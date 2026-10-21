"""
Dermora AI - classical morphometry.

A second, entirely independent opinion on the lesion, computed with deterministic
image processing rather than a neural network. This exists because a CNN's
posterior is opaque and because an examiner can check every number here by hand.

The pipeline is:

  1. DullRazor hair removal (Lee et al., 1997) - morphological blackhat, threshold,
     inpaint. Dermoscopy images are full of hair, and hair wrecks border metrics.
  2. Lesion segmentation - Otsu on the inverted L* channel inside a vignette mask,
     then largest central connected component with hole filling.
  3. The ABCD rule of dermoscopy (Stolz et al., 1994), scored as
     TDS = 1.3A + 0.1B + 0.5C + 0.5D, with the published interpretation bands
     (<4.75 benign, 4.75-5.45 suspicious, >5.45 highly suspicious).

Reported alongside, never instead of, the network output.
"""

from __future__ import annotations

import cv2
import numpy as np

# Reference colours for the C axis of the ABCD rule, in RGB. These are the six
# colours the published rule counts.
CLINICAL_COLORS = {
    "white": (255, 255, 255),
    "red": (204, 51, 51),
    "light-brown": (184, 134, 92),
    "dark-brown": (94, 60, 34),
    "blue-grey": (110, 125, 150),
    "black": (35, 30, 30),
}

MAX_WORK_DIM = 512          # segmentation runs on a downscaled copy for speed


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

def remove_hair(img_rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """
    DullRazor. Returns the inpainted image and the fraction of pixels that were
    identified as hair, which is itself a useful quality signal.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    k = max(5, (min(img_rgb.shape[:2]) // 60) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 12, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    frac = float(mask.mean() / 255.0)
    if frac < 0.002:                      # nothing worth inpainting
        return img_rgb, frac
    cleaned = cv2.inpaint(img_rgb, mask, 3, cv2.INPAINT_TELEA)
    return cleaned, frac


def _field_of_view(l_chan: np.ndarray) -> np.ndarray:
    """
    Mask of the usable optical field.

    Many dermoscopy images are shot down a contact lens and carry a black
    circular vignette. If that rim is left in, Otsu happily segments the vignette
    instead of the lesion and every downstream metric is garbage. So the field is
    derived from the image itself - the largest bright connected region, eroded
    slightly - and only falls back to a fixed ellipse when no vignette is found.
    """
    h, w = l_chan.shape[:2]
    bright = (l_chan > 32).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    n, labels, stats, _cent = cv2.connectedComponentsWithStats(bright, 8)
    if n > 1:
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        field = np.where(labels == idx, 255, 0).astype(np.uint8)
    else:
        field = bright

    if field.mean() / 255.0 < 0.25:                 # implausible - ignore it
        field = np.full((h, w), 255, np.uint8)

    # Pull in from the edge of the field so the soft vignette gradient is excluded.
    erode = max(3, int(min(h, w) * 0.035)) | 1
    field = cv2.erode(field, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode)))

    ellipse = np.zeros((h, w), np.uint8)
    cv2.ellipse(ellipse, (w // 2, h // 2), (int(w * 0.48), int(h * 0.48)), 0, 0, 360, 255, -1)
    return cv2.bitwise_and(field, ellipse)


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------

def segment_lesion(img_rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Returns a full-resolution uint8 mask (0/255) and a diagnostics dict.
    """
    h, w = img_rgb.shape[:2]
    scale = min(1.0, MAX_WORK_DIM / max(h, w))
    work = cv2.resize(img_rgb, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA) if scale < 1.0 else img_rgb.copy()
    wh, ww = work.shape[:2]

    blurred = cv2.GaussianBlur(work, (0, 0), sigmaX=max(1.0, min(wh, ww) / 150.0))
    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0]

    # Lesions are darker than surrounding skin: threshold the inverted luminance.
    inv = cv2.bitwise_not(l_chan)
    roi = _field_of_view(l_chan)
    roi_area = float((roi > 0).sum()) or 1.0
    vals = inv[roi > 0]
    thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(inv, thr, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, roi)

    # A lesion that appears to fill three-quarters of the optical field almost
    # always means Otsu split skin-vs-skin on a low-contrast image. Tighten the
    # threshold until the segment is plausible.
    for _ in range(4):
        if (mask > 0).sum() / roi_area <= 0.62:
            break
        thr = float(thr) + max(4.0, float(vals.std()) * 0.35)
        _, mask = cv2.threshold(inv, thr, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, roi)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Keep the component closest to the frame centre weighted by area - lesions
    # are photographed centrally by convention.
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    quality, chosen = "ok", None
    if n > 1:
        cx, cy = ww / 2, wh / 2
        best, best_score = None, -1.0
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < wh * ww * 0.004:
                continue
            dist = np.hypot(centroids[i][0] - cx, centroids[i][1] - cy) / np.hypot(cx, cy)
            score = (area / (wh * ww)) * (1.0 - 0.65 * dist)
            if score > best_score:
                best, best_score = i, score
        chosen = best

    if chosen is None:
        quality = "fallback"
        mask = np.zeros((wh, ww), np.uint8)
        cv2.ellipse(mask, (ww // 2, wh // 2), (int(ww * 0.30), int(wh * 0.30)),
                    0, 0, 360, 255, -1)
    else:
        mask = np.where(labels == chosen, 255, 0).astype(np.uint8)

    # Fill interior holes (regression areas, glare specks).
    ff = mask.copy()
    pad = np.zeros((wh + 2, ww + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 255)
    mask = mask | cv2.bitwise_not(ff)

    coverage = float(mask.mean() / 255.0)
    if coverage > 0.88 or coverage < 0.004:
        quality = "unreliable"

    if scale < 1.0:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask, {"coverage": coverage, "segmentation_quality": quality}


# --------------------------------------------------------------------------
# ABCD axes
# --------------------------------------------------------------------------

def _principal_align(mask: np.ndarray) -> np.ndarray:
    """Rotate the mask so its major axis is horizontal, for axis-fair asymmetry."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return mask
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    _vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, -1]
    angle = float(np.degrees(np.arctan2(major[1], major[0])))
    M = cv2.getRotationMatrix2D((float(mean[0]), float(mean[1])), angle, 1.0)
    return cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)


def asymmetry(mask: np.ndarray) -> dict:
    """
    Fractional non-overlap under reflection about each principal axis.
    A axis of the ABCD rule scores 0, 1 or 2 asymmetric axes.
    """
    aligned = _principal_align(mask)
    ys, xs = np.nonzero(aligned)
    if len(xs) < 10:
        return {"horizontal": 0.0, "vertical": 0.0, "index": 0.0, "score": 0}

    crop = aligned[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    area = float((crop > 0).sum()) or 1.0

    h_diff = float(np.logical_xor(crop > 0, np.fliplr(crop) > 0).sum()) / (2 * area)
    v_diff = float(np.logical_xor(crop > 0, np.flipud(crop) > 0).sum()) / (2 * area)

    thresh = 0.12
    score = int(h_diff > thresh) + int(v_diff > thresh)
    return {
        "horizontal": round(h_diff, 4),
        "vertical": round(v_diff, 4),
        "index": round((h_diff + v_diff) / 2, 4),
        "score": score,
    }


def border(mask: np.ndarray, img_rgb: np.ndarray) -> dict:
    """
    B axis: the lesion is divided into eight sectors about its centroid and each
    sector scores 1 if the pigment terminates abruptly at the border. Abruptness
    is measured as the luminance gradient across the boundary in that sector.
    """
    # Smooth the mask before measuring perimeter: a raw binary boundary is a
    # pixel staircase whose length is inflated by ~20-30%, which would make every
    # lesion look irregular.
    k = max(5, (min(mask.shape[:2]) // 45) | 1)
    smooth = cv2.GaussianBlur(mask, (k, k), 0)
    smooth = (smooth > 127).astype(np.uint8) * 255

    contours, _ = cv2.findContours(smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"score": 0, "irregularity": 0.0, "compactness": 0.0, "convexity": 0.0}

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    if perim > 0:
        cnt = cv2.approxPolyDP(cnt, 0.0025 * perim, True)
        perim = float(cv2.arcLength(cnt, True))
        area = float(cv2.contourArea(cnt))
    compactness = (4 * np.pi * area / (perim ** 2)) if perim > 0 else 0.0

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull)) or 1.0
    convexity = area / hull_area

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5),
                         cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5))

    M = cv2.moments(mask, binaryImage=True)
    if M["m00"] == 0:
        return {"score": 0, "irregularity": 0.0, "compactness": compactness,
                "convexity": convexity}
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

    pts = cnt.reshape(-1, 2).astype(np.float32)
    angles = (np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) + 360) % 360
    sector_of = (angles // 45).astype(int)

    sector_grads = []
    for s in range(8):
        sel = pts[sector_of == s]
        if len(sel) == 0:
            sector_grads.append(0.0)
            continue
        gy = np.clip(sel[:, 1].astype(int), 0, grad.shape[0] - 1)
        gx = np.clip(sel[:, 0].astype(int), 0, grad.shape[1] - 1)
        sector_grads.append(float(grad[gy, gx].mean()))

    ref = float(np.median([g for g in sector_grads if g > 0]) or 0.0)
    abrupt = sum(1 for g in sector_grads if ref > 0 and g >= ref * 1.10)

    return {
        "score": int(abrupt),
        "irregularity": round(float(1.0 - compactness), 4),
        "compactness": round(float(compactness), 4),
        "convexity": round(float(convexity), 4),
        "sector_gradients": [round(g, 2) for g in sector_grads],
    }


def colors(img_rgb: np.ndarray, mask: np.ndarray) -> dict:
    """
    C axis: count how many of the six clinical colours occupy a meaningful share
    of the lesion. Assignment is nearest-reference in CIELAB, which is
    perceptually uniform, rather than in RGB.
    """
    px = img_rgb[mask > 0]
    if len(px) < 30:
        return {"score": 1, "present": [], "shares": {}, "variegation": 0.0}

    if len(px) > 40000:                                 # subsample for speed
        px = px[np.random.default_rng(0).choice(len(px), 40000, replace=False)]

    lab_px = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    refs = np.array([[c] for c in CLINICAL_COLORS.values()], dtype=np.uint8)
    lab_refs = cv2.cvtColor(refs, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

    d = np.linalg.norm(lab_px[:, None, :] - lab_refs[None, :, :], axis=2)
    nearest = d.argmin(axis=1)

    names = list(CLINICAL_COLORS.keys())
    shares = {names[i]: float((nearest == i).mean()) for i in range(len(names))}
    present = [n for n, s in shares.items() if s >= 0.08]

    variegation = float(lab_px[:, 1:].std(axis=0).mean() / 30.0)

    return {
        "score": max(1, min(6, len(present))),
        "present": present,
        "shares": {k: round(v, 4) for k, v in sorted(shares.items(), key=lambda kv: -kv[1])},
        "variegation": round(min(variegation, 1.0), 4),
    }


def structures(img_rgb: np.ndarray, mask: np.ndarray) -> dict:
    """
    D axis approximation. The published rule counts five dermoscopic structures
    (network, structureless areas, dots, globules, streaks); identifying those
    reliably needs a dedicated model, so this substitutes the number of distinct
    colour-texture clusters inside the lesion, which correlates with structural
    heterogeneity. Reported as an approximation and labelled as such in the UI.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return {"score": 1, "clusters": 1, "heterogeneity": 0.0, "approximate": True}

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    texture = cv2.Laplacian(cv2.GaussianBlur(gray, (5, 5), 0), cv2.CV_32F)
    texture = cv2.GaussianBlur(np.abs(texture), (9, 9), 0)

    feats = np.column_stack([
        lab[ys, xs, 0].astype(np.float32),
        lab[ys, xs, 1].astype(np.float32) * 1.5,
        lab[ys, xs, 2].astype(np.float32) * 1.5,
        np.clip(texture[ys, xs], 0, 60).astype(np.float32) * 1.2,
    ])
    if len(feats) > 20000:
        feats = feats[np.random.default_rng(0).choice(len(feats), 20000, replace=False)]

    k = 5
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
    _compact, labels, centers = cv2.kmeans(feats, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k) / len(labels)

    # k-means always returns k clusters, so counting them directly would score
    # every lesion at 5. Merge clusters whose centroids are perceptually close
    # (CIELAB distance below the just-noticeable-difference used here) and count
    # only the merged groups that hold a meaningful share of the lesion.
    merged: list[list[int]] = []
    for i in np.argsort(-counts):
        for group in merged:
            if np.linalg.norm(centers[i][:3] - centers[group[0]][:3]) < 14.0:
                group.append(int(i))
                break
        else:
            merged.append([int(i)])

    shares = np.array([counts[g].sum() for g in merged])
    significant = int((shares >= 0.12).sum())

    heterogeneity = float(-(counts[counts > 0] * np.log(counts[counts > 0])).sum() / np.log(k))

    return {
        "score": max(1, min(5, significant)),
        "clusters": significant,
        "heterogeneity": round(heterogeneity, 4),
        "approximate": True,
    }


def geometry(mask: np.ndarray, img_shape: tuple[int, int]) -> dict:
    """D-for-diameter, in pixels and as a fraction of frame width."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"diameter_px": 0.0, "frame_fraction": 0.0, "area_fraction": 0.0}

    cnt = max(contours, key=cv2.contourArea)
    (_c, radius) = cv2.minEnclosingCircle(cnt)
    diameter = float(radius * 2)
    return {
        "diameter_px": round(diameter, 1),
        "frame_fraction": round(diameter / img_shape[1], 4),
        "area_fraction": round(float(cv2.contourArea(cnt)) / (img_shape[0] * img_shape[1]), 4),
        "note": "Absolute size in millimetres requires a calibrated scale in frame.",
    }


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def total_dermoscopy_score(a: int, b: int, c: int, d: int) -> dict:
    """Stolz ABCD rule: TDS = 1.3A + 0.1B + 0.5C + 0.5D."""
    tds = 1.3 * a + 0.1 * b + 0.5 * c + 0.5 * d
    if tds < 4.75:
        band, verdict = "BENIGN", "Below the 4.75 threshold - morphology favours a benign lesion."
    elif tds <= 5.45:
        band, verdict = "SUSPICIOUS", "In the 4.75-5.45 indeterminate band - morphology is equivocal."
    else:
        band, verdict = "HIGH", "Above 5.45 - morphology is highly suspicious for melanoma."
    return {"tds": round(float(tds), 2), "band": band, "verdict": verdict,
            "components": {"A": a, "B": b, "C": c, "D": d}}


def analyse(img_rgb: np.ndarray) -> dict:
    """Full morphometric workup. Never raises: degrades to a flagged result."""
    try:
        cleaned, hair_frac = remove_hair(img_rgb)
        mask, seg_info = segment_lesion(cleaned)

        a = asymmetry(mask)
        b = border(mask, cleaned)
        c = colors(cleaned, mask)
        d = structures(cleaned, mask)
        g = geometry(mask, img_rgb.shape[:2])

        tds = total_dermoscopy_score(a["score"], b["score"], c["score"], d["score"])

        return {
            "available": True,
            "hair_fraction": round(hair_frac, 4),
            "segmentation": seg_info,
            "asymmetry": a,
            "border": b,
            "color": c,
            "structures": d,
            "geometry": g,
            "abcd": tds,
            "evolution_note": (
                "The E axis (evolution) cannot be assessed from a single image. "
                "Serial photography at 3-6 month intervals is required."
            ),
            "_mask": mask,
        }
    except Exception as exc:                                   # pragma: no cover
        return {"available": False, "error": str(exc)}


def mask_outline(img_rgb: np.ndarray, mask: np.ndarray,
                 color: tuple[int, int, int] = (61, 220, 151)) -> np.ndarray:
    """Draw the segmentation boundary over the image for the UI."""
    out = img_rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thickness = max(2, int(min(img_rgb.shape[:2]) / 220))
    cv2.drawContours(out, contours, -1, color, thickness, lineType=cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
# image quality gate
# --------------------------------------------------------------------------

def assess_quality(img_rgb: np.ndarray) -> dict:
    """
    Cheap sanity checks run before the model is trusted. The point is to avoid
    confidently classifying a photograph of a keyboard, and to warn when an
    image is too blurred or too dark for the prediction to mean anything.
    """
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Laplacian variance scales with image size, so normalise to a fixed working
    # width first - otherwise a sharp 4000px phone photo and a sharp 320px crop
    # get wildly different focus scores.
    fw = 512
    focus_src = cv2.resize(gray, (fw, max(1, int(h * fw / w))),
                           interpolation=cv2.INTER_AREA) if w != fw else gray
    focus = float(cv2.Laplacian(focus_src, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    clipped_hi = float((gray > 250).mean())
    clipped_lo = float((gray < 5).mean())

    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin = ((cr > 128) & (cr < 190) & (cb > 75) & (cb < 138) & (gray > 25))
    skin_fraction = float(skin.mean())

    flags: list[str] = []
    if min(h, w) < 128:
        flags.append("Image resolution is below 128px; fine dermoscopic detail is unrecoverable.")
    if focus < 45:
        flags.append("Image appears out of focus; border and texture metrics will be unreliable.")
    if brightness < 45:
        flags.append("Image is underexposed.")
    elif brightness > 215:
        flags.append("Image is overexposed.")
    if clipped_hi > 0.12:
        flags.append("Significant specular glare detected.")
    if contrast < 18:
        flags.append("Very low contrast; the lesion may not be separable from surrounding skin.")
    if skin_fraction < 0.25:
        flags.append("Little skin-toned content detected - this may not be a skin lesion image.")

    score = 100.0
    score -= max(0.0, (60 - min(focus, 60)) * 0.55)
    score -= abs(brightness - 130) * 0.16
    score -= clipped_hi * 110
    score -= max(0.0, (0.45 - min(skin_fraction, 0.45))) * 95
    score -= max(0.0, (25 - min(contrast, 25))) * 0.9
    score = float(np.clip(score, 0, 100))

    return {
        "score": round(score, 1),
        "focus": round(focus, 1),
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "glare_fraction": round(clipped_hi, 4),
        "dark_fraction": round(clipped_lo, 4),
        "skin_fraction": round(skin_fraction, 4),
        "resolution": [int(w), int(h)],
        "flags": flags,
        "usable": score >= 40 and skin_fraction >= 0.15,
    }
