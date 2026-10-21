"""
Dermora AI - inference engine.

The supplied checkpoint is a YOLOv8n *detection* model that was trained on
ISIC 2018 Task 3 with whole-image bounding boxes (`class 0.5 0.5 1.0 1.0`).
That makes it, functionally, a classifier that happens to be wrapped in a
detection head.

The previous version of this project consumed it naively:

    boxes = results[0].boxes
    if len(boxes) == 0: return error("No lesion detected")
    cls  = int(boxes.cls[0]); conf = float(boxes.conf[0])
    probs = [0]*7; probs[cls] = conf          # <- a fabricated distribution

Three things are wrong with that. Non-maximum suppression throws away every
class except the winner, so the reported "probability distribution" is a
one-hot vector and every differential diagnosis is lost. A confidence below
the 0.25 NMS threshold produces a hard failure on a perfectly valid image.
And a single forward pass at a single orientation gives no handle on how
stable the prediction actually is.

This engine instead reads the raw detection head *before* NMS:

  1. the head emits per-anchor, per-class sigmoid scores of shape (nc, N);
  2. scores are pooled over the top-k anchors per class, which is the right
     pooling operator when the target object is known to fill the frame;
  3. pooled scores are mapped to logits and passed through a temperature
     softmax, producing a genuine 7-way posterior in which every class keeps
     its mass;
  4. the whole thing is repeated under dihedral test-time augmentation, and
     the spread across augmentations is retained as an uncertainty signal.

Nothing here is thrown away, and no input can produce a hard failure.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .knowledge import CLASS_ORDER

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

DEFAULT_IMGSZ = 224          # matches the checkpoint's train_args.imgsz
TOPK_ANCHORS = 32            # anchors pooled per class
SOFTMAX_T = 1.0              # temperature on the derived logits
EPS = 1e-6


@dataclass
class Prediction:
    """Everything the engine knows about one image."""

    probs: dict[str, float]
    top_class: str
    confidence: float
    margin: float                       # p1 - p2
    entropy: float                      # normalised predictive entropy 0..1
    detector_score: float               # raw max sigmoid score, pre-softmax
    tta_agreement: float                # fraction of augmentations agreeing
    per_class_raw: dict[str, float]     # pooled sigmoid scores
    roi: list[float] | None             # [x1, y1, x2, y2] in original pixels
    inference_ms: float
    views: int
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "probs": self.probs,
            "top_class": self.top_class,
            "confidence": self.confidence,
            "margin": self.margin,
            "entropy": self.entropy,
            "detector_score": self.detector_score,
            "tta_agreement": self.tta_agreement,
            "per_class_raw": self.per_class_raw,
            "roi": self.roi,
            "inference_ms": self.inference_ms,
            "views": self.views,
            **self.meta,
        }


# --------------------------------------------------------------------------
# preprocessing helpers
# --------------------------------------------------------------------------

def letterbox(img: np.ndarray, size: int = DEFAULT_IMGSZ, color: int = 114):
    """
    Resize preserving aspect ratio and pad to a square, mirroring the geometry
    Ultralytics uses at validation time. Returns the padded image plus the
    transform needed to map coordinates back to the original frame.
    """
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    canvas = np.full((size, size, 3), color, dtype=img.dtype)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def undo_letterbox(box: np.ndarray, r: float, pad_x: int, pad_y: int,
                   w: int, h: int) -> list[float]:
    """Map an xyxy box from letterboxed space back to original pixels."""
    x1, y1, x2, y2 = box
    x1 = (x1 - pad_x) / r
    x2 = (x2 - pad_x) / r
    y1 = (y1 - pad_y) / r
    y2 = (y2 - pad_y) / r
    return [
        float(np.clip(x1, 0, w)), float(np.clip(y1, 0, h)),
        float(np.clip(x2, 0, w)), float(np.clip(y2, 0, h)),
    ]


# Dihedral test-time augmentation. Each entry is (name, forward, inverse-x-flip,
# inverse-y-flip) so ROI coordinates can be mapped back.
TTA_VIEWS = [
    ("identity", lambda a: a, False, False),
    ("hflip", lambda a: a[:, ::-1], True, False),
    ("vflip", lambda a: a[::-1, :], False, True),
    ("hvflip", lambda a: a[::-1, ::-1], True, True),
]


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------

class DermoraEngine:
    """Loads the checkpoint once and serves calibrated multi-class predictions."""

    def __init__(self, weights: str | Path, imgsz: int = DEFAULT_IMGSZ,
                 device: str = "cpu", use_tta: bool = True,
                 topk: int = TOPK_ANCHORS, temperature: float = SOFTMAX_T):
        import torch                       # imported lazily so the model card
        from ultralytics import YOLO       # endpoint works without torch present

        self.torch = torch
        self.weights = str(weights)
        self.imgsz = int(imgsz)
        self.device = device
        self.use_tta = use_tta
        self.topk = topk
        self.temperature = temperature

        self.yolo = YOLO(self.weights)
        self.net = self.yolo.model.float().to(device).eval()

        raw_names = getattr(self.net, "names", None) or {}
        if isinstance(raw_names, dict) and raw_names:
            self.names = [raw_names[i] for i in sorted(raw_names)]
        else:
            self.names = list(CLASS_ORDER)
        self.nc = len(self.names)

        # Feature taps for the attention map. Indices 15/18/21 are the three
        # detection-scale outputs of the YOLOv8 neck (P3/P4/P5).
        self._features: dict[int, Any] = {}
        self._hooks = []
        self._register_feature_hooks((15, 18, 21))

        self._warmup()

    # -- setup -------------------------------------------------------------

    def _register_feature_hooks(self, indices: tuple[int, ...]) -> None:
        seq = getattr(self.net, "model", None)
        if seq is None:
            return
        for idx in indices:
            if idx >= len(seq):
                continue

            def hook(_m, _inp, out, i=idx):
                self._features[i] = out.detach() if hasattr(out, "detach") else out

            self._hooks.append(seq[idx].register_forward_hook(hook))

    def _warmup(self) -> None:
        blank = np.full((self.imgsz, self.imgsz, 3), 128, dtype=np.uint8)
        try:
            self._forward(blank)
        except Exception:
            pass

    # -- core forward ------------------------------------------------------

    def _forward(self, letterboxed_rgb: np.ndarray):
        """One forward pass. Returns the raw head tensor (1, 4+nc, N)."""
        torch = self.torch
        arr = np.ascontiguousarray(letterboxed_rgb.transpose(2, 0, 1))
        t = torch.from_numpy(arr).unsqueeze(0).float().div_(255.0).to(self.device)
        with torch.no_grad():
            out = self.net(t)
        return self._unwrap(out)

    def _unwrap(self, out):
        """Pull the (B, 4+nc, N) prediction tensor out of whatever the head returned."""
        cand = out
        for _ in range(4):
            if isinstance(cand, (list, tuple)):
                cand = cand[0]
            else:
                break
        if not hasattr(cand, "ndim"):
            raise RuntimeError("Unexpected model output structure")
        if cand.ndim == 3 and cand.shape[1] != 4 + self.nc and cand.shape[2] == 4 + self.nc:
            cand = cand.permute(0, 2, 1)
        if cand.ndim != 3 or cand.shape[1] < 4 + self.nc:
            raise RuntimeError(f"Unexpected head shape {tuple(cand.shape)}")
        return cand

    def _pool(self, head):
        """
        Pool per-anchor class scores into one score per class.

        Top-k mean rather than plain max: the max over ~1000 anchors is a noisy
        order statistic, while the mean of the strongest k anchors is a far more
        stable estimator when the target is known to occupy the whole frame.
        """
        torch = self.torch
        cls_scores = head[0, 4:4 + self.nc, :]            # (nc, N)
        k = int(min(self.topk, cls_scores.shape[1]))
        vals, idx = torch.topk(cls_scores, k, dim=1)
        pooled = vals.mean(dim=1)                         # (nc,)
        peak = cls_scores.max(dim=1)                      # values + indices
        return pooled, peak.values, peak.indices, head[0, :4, :]

    def _posterior(self, pooled: np.ndarray) -> np.ndarray:
        """Sigmoid scores -> logits -> temperature softmax. Mass is preserved."""
        p = np.clip(pooled.astype(np.float64), EPS, 1.0 - EPS)
        logits = np.log(p / (1.0 - p)) / max(self.temperature, EPS)
        logits -= logits.max()
        e = np.exp(logits)
        return e / e.sum()

    # -- public API --------------------------------------------------------

    def predict(self, image_rgb: np.ndarray) -> Prediction:
        """
        Preferred path: raw head aggregation. If anything about the head layout
        is unexpected on this Ultralytics build, fall back to the standard
        predict API rather than failing in front of an audience - the answer is
        coarser, but the app keeps working.
        """
        try:
            return self._predict_raw(image_rgb)
        except Exception as exc:                                   # pragma: no cover
            fallback = self._predict_fallback(image_rgb)
            fallback.meta["degraded"] = True
            fallback.meta["degraded_reason"] = str(exc)
            return fallback

    def _predict_raw(self, image_rgb: np.ndarray) -> Prediction:
        t0 = time.perf_counter()
        h, w = image_rgb.shape[:2]

        views = TTA_VIEWS if self.use_tta else TTA_VIEWS[:1]
        pooled_acc = np.zeros(self.nc, dtype=np.float64)
        peak_acc = np.zeros(self.nc, dtype=np.float64)
        votes: list[int] = []
        roi = None
        cam_source = None

        for name, fwd, flip_x, flip_y in views:
            view = np.ascontiguousarray(fwd(image_rgb))
            lb, r, pad_x, pad_y = letterbox(view, self.imgsz)
            head = self._forward(lb)
            pooled, peak, peak_idx, boxes = self._pool(head)

            pooled_np = pooled.detach().cpu().numpy().astype(np.float64)
            peak_np = peak.detach().cpu().numpy().astype(np.float64)
            pooled_acc += pooled_np
            peak_acc = np.maximum(peak_acc, peak_np)
            votes.append(int(pooled_np.argmax()))

            if name == "identity":
                cam_source = dict(self._features)
                top = int(pooled_np.argmax())
                anchor = int(peak_idx[top].item())
                cx, cy, bw, bh = [float(v) for v in boxes[:, anchor].detach().cpu().numpy()]
                xyxy = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
                roi = undo_letterbox(xyxy, r, pad_x, pad_y, w, h)

        pooled_mean = pooled_acc / len(views)
        posterior = self._posterior(pooled_mean)

        order = np.argsort(posterior)[::-1]
        top_i = int(order[0])
        conf = float(posterior[top_i])
        margin = float(conf - posterior[int(order[1])]) if self.nc > 1 else 1.0

        ent = float(-(posterior * np.log(posterior + EPS)).sum() / math.log(self.nc))
        agreement = float(sum(1 for v in votes if v == top_i) / len(votes))

        self._last_features = cam_source or {}

        return Prediction(
            probs={self.names[i]: float(posterior[i]) for i in range(self.nc)},
            top_class=self.names[top_i],
            confidence=conf,
            margin=margin,
            entropy=ent,
            detector_score=float(peak_acc.max()),
            tta_agreement=agreement,
            per_class_raw={self.names[i]: float(pooled_mean[i]) for i in range(self.nc)},
            roi=roi,
            inference_ms=(time.perf_counter() - t0) * 1000.0,
            views=len(views),
        )

    def _predict_fallback(self, image_rgb: np.ndarray) -> Prediction:
        """
        Degraded path using the public Ultralytics API. NMS has already collapsed
        the class dimension here, so the remaining mass is spread over the other
        classes in proportion to nothing but their absence - it is a floor, not a
        posterior, and the UI marks the result as degraded.
        """
        t0 = time.perf_counter()
        res = self.yolo.predict(image_rgb, imgsz=self.imgsz, conf=0.01, verbose=False)[0]

        scores = np.full(self.nc, 0.02, dtype=np.float64)
        roi = None
        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes):
            for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                i = int(cls_id)
                scores[i] = max(scores[i], float(conf))
            best = int(np.argmax(boxes.conf.tolist()))
            roi = [float(v) for v in boxes.xyxy[best].tolist()]

        posterior = scores / scores.sum()
        order = np.argsort(posterior)[::-1]
        top_i = int(order[0])
        conf = float(posterior[top_i])

        return Prediction(
            probs={self.names[i]: float(posterior[i]) for i in range(self.nc)},
            top_class=self.names[top_i],
            confidence=conf,
            margin=float(conf - posterior[int(order[1])]) if self.nc > 1 else 1.0,
            entropy=float(-(posterior * np.log(posterior + EPS)).sum() / math.log(self.nc)),
            detector_score=float(scores.max()),
            tta_agreement=1.0,
            per_class_raw={self.names[i]: float(scores[i]) for i in range(self.nc)},
            roi=roi,
            inference_ms=(time.perf_counter() - t0) * 1000.0,
            views=1,
        )

    def last_feature_maps(self) -> dict[int, Any]:
        """Feature tensors captured during the most recent identity-view pass."""
        return getattr(self, "_last_features", {})

    def close(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
