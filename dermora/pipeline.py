"""
Dermora AI - analysis pipeline.

Ties the four subsystems together into one case record:

    quality gate -> network posterior -> Eigen-CAM -> classical morphometry

and produces the JSON contract the front end and the PDF reporter both consume.
Everything is defensive: a failure in explainability or morphometry degrades
that section only, it never fails the case.
"""

from __future__ import annotations

import base64
import threading
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import explain, morphology
from .engine import DermoraEngine
from .knowledge import (LESIONS, RISK_TIERS, concerning_probability,
                        malignant_probability, narrative, triage_band)

DISPLAY_MAX = 760          # longest edge of images returned to the browser
THUMB_MAX = 168

DISCLAIMER = (
    "Dermora AI is an academic decision-support prototype trained on ISIC 2018 Task 3 "
    "(HAM10000). It is not a medical device, carries no regulatory clearance, and must "
    "not be used to make or defer treatment decisions. Histopathology remains the "
    "diagnostic standard."
)


def _encode(img_rgb: np.ndarray, max_dim: int = DISPLAY_MAX, quality: int = 90) -> bytes:
    h, w = img_rgb.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def _data_url(raw: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def _draw_roi(img_rgb: np.ndarray, roi: list[float] | None) -> np.ndarray:
    if not roi:
        return img_rgb
    out = img_rgb.copy()
    x1, y1, x2, y2 = [int(v) for v in roi]
    if x2 - x1 < 4 or y2 - y1 < 4:
        return out
    t = max(2, int(min(out.shape[:2]) / 200))
    cv2.rectangle(out, (x1, y1), (x2, y2), (14, 220, 220), t, lineType=cv2.LINE_AA)
    return out


class Pipeline:
    """Thread-safe wrapper around the engine plus the analysis stages."""

    def __init__(self, weights: str | Path, imgsz: int = 224, use_tta: bool = True):
        self.engine = DermoraEngine(weights, imgsz=imgsz, use_tta=use_tta)
        self.imgsz = imgsz
        self._lock = threading.Lock()          # torch modules are not re-entrant

    # ---------------------------------------------------------------- public

    def analyse(self, img_rgb: np.ndarray, filename: str = "upload.jpg",
                with_morphology: bool = True) -> tuple[dict, dict[str, bytes]]:
        started = datetime.now()
        case_id = f"DRM-{uuid.uuid4().hex[:8].upper()}"

        quality = morphology.assess_quality(img_rgb)

        with self._lock:
            pred = self.engine.predict(img_rgb)
            features = self.engine.last_feature_maps()
            overlay, cam = explain.attention_overlay(img_rgb, features, self.imgsz)

        att_stats = explain.attention_stats(cam, pred.roi)

        morph: dict = {"available": False}
        outline_img = None
        if with_morphology:
            morph = morphology.analyse(img_rgb)
            mask = morph.pop("_mask", None)
            if mask is not None:
                outline_img = morphology.mask_outline(img_rgb, mask)

        probs = pred.probs
        concern = concerning_probability(probs)
        malignant = malignant_probability(probs)
        triage = triage_band(concern, pred.top_class, pred.confidence)

        # A poor-quality image must not be allowed to produce a confident-looking
        # answer; the triage band is downgraded and the reason is surfaced.
        caveats: list[str] = list(quality.get("flags", []))
        if not quality.get("usable", True):
            caveats.append(
                "Image quality is below the threshold at which this model's output is "
                "meaningful. Re-capture before acting on the result."
            )
        if pred.tta_agreement < 1.0:
            caveats.append(
                f"Only {pred.tta_agreement * 100:.0f}% of augmented views agreed on the "
                "leading class - the prediction is orientation-sensitive."
            )
        if pred.margin < 0.10:
            caveats.append(
                "The top two classes are separated by under 10 percentage points; treat "
                "this as an indeterminate result."
            )
        if morph.get("available") and morph["abcd"]["band"] != "BENIGN" and \
                triage["band"] in ("ROUTINE", "REVIEW"):
            caveats.append(
                f"Morphometry disagrees with the network: the ABCD total dermoscopy score "
                f"is {morph['abcd']['tds']} ({morph['abcd']['band'].lower()}) while the "
                f"network favours a benign class. Human review is advised."
            )

        ranked = [
            {
                "code": code,
                "name": LESIONS[code]["name"],
                "category": LESIONS[code]["category"],
                "risk": LESIONS[code]["risk"],
                "color": RISK_TIERS[LESIONS[code]["risk"]]["color"],
                "prob": float(p),
            }
            for code, p in sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        ]

        top_rec = LESIONS[pred.top_class]
        differentials = [
            {
                "code": c,
                "name": LESIONS[c]["name"],
                "prob": float(probs.get(c, 0.0)),
                "category": LESIONS[c]["category"],
                "risk": LESIONS[c]["risk"],
                "distinguisher": LESIONS[c]["dermoscopy"][0],
            }
            for c in top_rec["differentials"]
        ]

        images = {
            "original": _encode(img_rgb),
            "attention": _encode(overlay) if overlay is not None else b"",
            "outline": _encode(outline_img) if outline_img is not None else b"",
            "roi": _encode(_draw_roi(img_rgb, pred.roi)),
            "thumb": _encode(img_rgb, THUMB_MAX, 78),
        }

        case = {
            "case_id": case_id,
            "filename": filename,
            "timestamp": started.isoformat(timespec="seconds"),
            "elapsed_ms": round((datetime.now() - started).total_seconds() * 1000, 1),
            "quality": quality,
            "prediction": {
                **pred.as_dict(),
                "ranked": ranked,
                "top_name": top_rec["name"],
                "top_full_name": top_rec["full_name"],
                "top_category": top_rec["category"],
            },
            "triage": triage,
            "concern": concern,
            "malignant_probability": malignant,
            "lesion": top_rec,
            "differentials": differentials,
            "morphology": morph,
            "attention": att_stats,
            "caveats": caveats,
            "narrative": narrative(pred.top_class, pred.confidence, pred.margin, probs),
            "disclaimer": DISCLAIMER,
        }
        return case, images

    def as_payload(self, case: dict, images: dict[str, bytes]) -> dict:
        """Case dict plus inline images, ready to hand to the browser."""
        return {
            **case,
            "images": {k: _data_url(v) for k, v in images.items() if v},
        }
