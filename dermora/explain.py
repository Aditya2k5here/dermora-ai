"""
Dermora AI - visual explainability.

Implements Eigen-CAM (Muhammad & Yeasin, IJCNN 2020) over the three neck
outputs of the YOLOv8 backbone. Eigen-CAM projects the activation tensor onto
its first principal component, which yields a class-agnostic saliency map
without needing gradients. That matters here for two practical reasons: the
detection head is post-NMS-free in our pipeline so there is no single scalar to
differentiate, and a gradient-free method runs comfortably on CPU inside a live
demo.

The three feature scales are combined rather than used individually: P3
(stride 8) localises the lesion border, P5 (stride 32) captures global context,
and averaging the normalised maps gives a saliency image that is stable across
lesion sizes.
"""

from __future__ import annotations

import cv2
import numpy as np

# Relative weights for the P3 / P4 / P5 taps registered in engine.py.
SCALE_WEIGHTS = {15: 0.5, 18: 0.3, 21: 0.2}


def _eigen_map(feat: np.ndarray) -> np.ndarray:
    """
    First principal component of a (C, H, W) activation tensor, returned as a
    normalised (H, W) saliency map.
    """
    c, h, w = feat.shape
    flat = feat.reshape(c, h * w).T                      # (HW, C)
    flat = flat - flat.mean(axis=0, keepdims=True)

    try:
        # Only the leading singular vector is needed.
        _u, _s, vt = np.linalg.svd(flat, full_matrices=False)
        proj = flat @ vt[0]
    except np.linalg.LinAlgError:
        proj = flat.mean(axis=1)

    cam = proj.reshape(h, w)
    # Sign of a principal component is arbitrary; orient it so the lesion
    # (the high-energy region) is positive.
    if cam.mean() < 0:
        cam = -cam
    cam = np.maximum(cam, 0)
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def saliency_from_features(features: dict, out_size: tuple[int, int]) -> np.ndarray | None:
    """
    Combine the captured neck features into a single float32 map in [0, 1]
    sized (H, W) = out_size.
    """
    if not features:
        return None

    acc, total = None, 0.0
    for idx, tensor in features.items():
        try:
            arr = tensor[0].detach().cpu().numpy().astype(np.float32)
        except Exception:
            continue
        if arr.ndim != 3 or arr.shape[1] < 2 or arr.shape[2] < 2:
            continue

        cam = _eigen_map(arr)
        cam = cv2.resize(cam, (out_size[1], out_size[0]), interpolation=cv2.INTER_CUBIC)
        wgt = SCALE_WEIGHTS.get(idx, 0.2)
        acc = cam * wgt if acc is None else acc + cam * wgt
        total += wgt

    if acc is None or total == 0:
        return None

    acc /= total
    acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=max(out_size) / 90.0)
    lo, hi = float(acc.min()), float(acc.max())
    return (acc - lo) / (hi - lo) if hi > lo else np.zeros_like(acc)


def _letterbox_crop(cam: np.ndarray, orig_h: int, orig_w: int, imgsz: int) -> np.ndarray:
    """
    The network saw a letterboxed square. Strip the grey padding bands from the
    saliency map before stretching it back over the original frame, otherwise
    the heatmap is spatially offset from the lesion.
    """
    r = min(imgsz / orig_h, imgsz / orig_w)
    nh, nw = int(round(orig_h * r)), int(round(orig_w * r))
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    cam = cv2.resize(cam, (imgsz, imgsz), interpolation=cv2.INTER_CUBIC)
    return cam[top:top + nh, left:left + nw]


def attention_overlay(image_rgb: np.ndarray, features: dict, imgsz: int,
                      alpha: float = 0.45) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Returns (overlay_rgb, raw_cam_float) for the supplied image.
    """
    h, w = image_rgb.shape[:2]
    cam = saliency_from_features(features, (imgsz, imgsz))
    if cam is None:
        return None, None

    cam = _letterbox_crop(cam, h, w, imgsz)
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam = np.clip(cam, 0, 1)

    # Gamma pushes the map towards its hot core so the overlay reads clearly on
    # a projector rather than washing the whole image in orange.
    shaped = np.power(cam, 1.6)
    heat = cv2.applyColorMap((shaped * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32)

    weight = (shaped[..., None] * alpha).astype(np.float32)
    blended = image_rgb.astype(np.float32) * (1 - weight) + heat * weight
    return np.clip(blended, 0, 255).astype(np.uint8), cam


def attention_stats(cam: np.ndarray | None, roi: list[float] | None) -> dict:
    """
    Quantify where the model looked. `concentration` is the share of total
    saliency inside the top-decile region: high values mean the model committed
    to a focal area, low values mean diffuse attention that deserves scepticism.
    """
    if cam is None:
        return {"available": False}

    flat = cam.ravel()
    thresh = float(np.quantile(flat, 0.90))
    hot = flat >= thresh
    concentration = float(flat[hot].sum() / (flat.sum() + 1e-8))

    ys, xs = np.where(cam >= thresh)
    centroid = None
    if len(xs):
        centroid = [float(xs.mean() / cam.shape[1]), float(ys.mean() / cam.shape[0])]

    off_centre = None
    if centroid:
        off_centre = float(np.hypot(centroid[0] - 0.5, centroid[1] - 0.5))

    return {
        "available": True,
        "concentration": concentration,
        "centroid": centroid,
        "off_centre": off_centre,
        "interpretation": (
            "Attention is focal and centred on the lesion."
            if concentration >= 0.30 and (off_centre or 0) < 0.25
            else "Attention is diffuse or off-centre; interpret the prediction with caution."
        ),
    }
