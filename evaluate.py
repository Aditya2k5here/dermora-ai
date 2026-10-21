"""
Dermora AI - quantitative evaluation.

Measures top-1 accuracy of the Dermora inference path on a labelled YOLO-format
split, and - the point of the exercise - compares it head to head with the
naive post-NMS `boxes.cls[0]` approach the previous version of this project
used, on identical images with identical weights.

    python evaluate.py --images D:/yolov8/dataset/images/val \
                       --labels D:/yolov8/dataset/labels/val

Writes evaluation.json and confusion_matrix.png next to this file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from dermora.engine import DermoraEngine            # noqa: E402
from dermora.knowledge import CLASS_ORDER, MALIGNANT  # noqa: E402

IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp"}


# --------------------------------------------------------------------------

def load_split(images: Path, labels: Path, limit: int | None):
    pairs = []
    for img in sorted(images.iterdir()):
        if img.suffix.lower() not in IMG_SUFFIX:
            continue
        lab = labels / f"{img.stem}.txt"
        if not lab.exists():
            continue
        try:
            cls_id = int(lab.read_text().split()[0])
        except (ValueError, IndexError):
            continue
        pairs.append((img, cls_id))
        if limit and len(pairs) >= limit:
            break
    return pairs


def metrics_from_confusion(cm: np.ndarray, names: list[str]) -> dict:
    total = cm.sum()
    correct = np.trace(cm)
    per_class = {}
    f1s, recalls = [], []

    for i, name in enumerate(names):
        tp = cm[i, i]
        support = cm[i].sum()
        predicted = cm[:, i].sum()
        recall = tp / support if support else float("nan")
        precision = tp / predicted if predicted else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if support and (precision + recall) else 0.0)
        per_class[name] = {
            "support": int(support),
            "recall": None if np.isnan(recall) else round(float(recall), 4),
            "precision": round(float(precision), 4),
            "f1": round(float(f1), 4),
        }
        if support:
            recalls.append(recall)
            f1s.append(f1)

    return {
        "accuracy": round(float(correct / total), 4) if total else 0.0,
        "balanced_accuracy": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "macro_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "n": int(total),
        "per_class": per_class,
    }


def malignant_sensitivity(truths: list[int], preds: list[int], names: list[str]) -> float:
    """Fraction of truly malignant lesions predicted as some malignant class."""
    mal = {i for i, n in enumerate(names) if n in MALIGNANT}
    hits = [1 for t, p in zip(truths, preds) if t in mal and p in mal]
    total = sum(1 for t in truths if t in mal)
    return round(len(hits) / total, 4) if total else float("nan")


def draw_confusion(cm: np.ndarray, names: list[str], title: str, path: Path) -> None:
    """Render a confusion matrix to PNG with OpenCV (no matplotlib dependency)."""
    n = len(names)
    cell, pad_l, pad_t = 76, 96, 78
    W, H = pad_l + n * cell + 30, pad_t + n * cell + 60
    img = np.full((H, W, 3), 12, np.uint8)

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img, title, (pad_l - 60, 34), font, 0.62, (232, 241, 248), 1, cv2.LINE_AA)

    for i in range(n):
        for j in range(n):
            v = norm[i, j]
            colour = (int(40 + 150 * v), int(30 + 190 * v), int(20 + 60 * v))  # BGR teal ramp
            if i == j:
                colour = (int(60 + 90 * v), int(60 + 175 * v), int(40 + 60 * v))
            x, y = pad_l + j * cell, pad_t + i * cell
            cv2.rectangle(img, (x, y), (x + cell - 3, y + cell - 3), colour, -1)
            txt = f"{cm[i, j]}"
            ink = (255, 255, 255) if v > 0.35 else (150, 168, 184)
            cv2.putText(img, txt, (x + cell // 2 - 6 * len(txt), y + cell // 2 + 5),
                        font, 0.46, ink, 1, cv2.LINE_AA)

    for i, name in enumerate(names):
        cv2.putText(img, name, (10, pad_t + i * cell + cell // 2 + 5),
                    font, 0.46, (170, 190, 205), 1, cv2.LINE_AA)
        cv2.putText(img, name, (pad_l + i * cell + 8, pad_t - 12),
                    font, 0.44, (170, 190, 205), 1, cv2.LINE_AA)

    cv2.putText(img, "rows = ground truth   columns = prediction", (pad_l - 60, H - 22),
                font, 0.44, (110, 130, 145), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate Dermora AI against a labelled split")
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--weights", type=Path, default=ROOT / "model" / "best.pt")
    ap.add_argument("--limit", type=int, default=0, help="cap the number of images")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="skip the post-NMS comparison run")
    args = ap.parse_args()

    pairs = load_split(args.images, args.labels, args.limit or None)
    if not pairs:
        print(f"No image/label pairs found under {args.images}")
        return 1
    print(f"Evaluating {len(pairs)} images from {args.images}")

    engine = DermoraEngine(args.weights, imgsz=args.imgsz, use_tta=not args.no_tta)
    names = engine.names
    n = len(names)

    cm_new = np.zeros((n, n), int)
    cm_old = np.zeros((n, n), int)
    truths, preds_new, preds_old = [], [], []
    old_no_detection = 0
    confidences, correctness = [], []
    t0 = time.perf_counter()

    for k, (path, truth) in enumerate(pairs, 1):
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

        pred = engine.predict(img)
        p_new = names.index(pred.top_class)
        cm_new[truth, p_new] += 1
        truths.append(truth)
        preds_new.append(p_new)
        confidences.append(pred.confidence)
        correctness.append(1 if p_new == truth else 0)

        if not args.skip_baseline:
            # Exactly what the previous app did: single pass, NMS at conf 0.25,
            # take the first surviving box.
            res = engine.yolo.predict(img, imgsz=args.imgsz, conf=0.25, verbose=False)[0]
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                old_no_detection += 1
                p_old = -1                      # the old UI showed an error here
            else:
                p_old = int(boxes.cls[0])
                cm_old[truth, p_old] += 1
            preds_old.append(p_old)

        if k % 25 == 0 or k == len(pairs):
            print(f"  {k}/{len(pairs)}  ({(time.perf_counter() - t0) / k * 1000:.0f} ms/image)")

    elapsed = time.perf_counter() - t0

    new = metrics_from_confusion(cm_new, names)
    new["malignant_sensitivity"] = malignant_sensitivity(truths, preds_new, names)

    report = {
        "images": len(pairs),
        "seconds": round(elapsed, 1),
        "ms_per_image": round(elapsed / len(pairs) * 1000, 1),
        "tta_views": pred.views,
        "dermora": new,
    }

    print("\n" + "=" * 62)
    print("  DERMORA PIPELINE (raw head + top-k pooling + TTA)")
    print("=" * 62)
    print(f"  top-1 accuracy      {new['accuracy'] * 100:6.2f}%")
    print(f"  balanced accuracy   {new['balanced_accuracy'] * 100:6.2f}%")
    print(f"  macro F1            {new['macro_f1'] * 100:6.2f}%")
    print(f"  melanoma+BCC recall {new['malignant_sensitivity'] * 100:6.2f}%")
    print(f"\n  {'class':<8}{'support':>9}{'recall':>10}{'precision':>12}{'f1':>8}")
    for name, m in new["per_class"].items():
        r = "  n/a" if m["recall"] is None else f"{m['recall'] * 100:6.1f}%"
        print(f"  {name:<8}{m['support']:>9}{r:>10}{m['precision'] * 100:>11.1f}%{m['f1'] * 100:>7.1f}%")

    if not args.skip_baseline:
        old_correct = sum(1 for t, p in zip(truths, preds_old) if t == p)
        old = metrics_from_confusion(cm_old, names)
        old["accuracy_including_failures"] = round(old_correct / len(pairs), 4)
        old["no_detection_rate"] = round(old_no_detection / len(pairs), 4)
        old["malignant_sensitivity"] = malignant_sensitivity(
            truths, [p if p >= 0 else -1 for p in preds_old], names)
        report["baseline_post_nms"] = old
        report["delta_accuracy_points"] = round(
            (new["accuracy"] - old["accuracy_including_failures"]) * 100, 2)

        print("\n" + "=" * 62)
        print("  BASELINE (post-NMS argmax, conf 0.25 - the previous approach)")
        print("=" * 62)
        print(f"  top-1 accuracy      {old['accuracy_including_failures'] * 100:6.2f}%")
        print(f"  balanced accuracy   {old['balanced_accuracy'] * 100:6.2f}%")
        print(f"  'no lesion detected' hard failures: {old_no_detection} "
              f"({old['no_detection_rate'] * 100:.1f}% of images)")
        print(f"\n  Dermora is {report['delta_accuracy_points']:+.2f} accuracy points on this split, "
              f"with zero hard failures.")

    # Calibration: mean confidence vs observed accuracy in three bands.
    bands = {"low (<50%)": [], "mid (50-75%)": [], "high (>75%)": []}
    for c, ok in zip(confidences, correctness):
        key = "low (<50%)" if c < 0.5 else "mid (50-75%)" if c < 0.75 else "high (>75%)"
        bands[key].append(ok)
    report["calibration"] = {
        k: {"n": len(v), "accuracy": round(float(np.mean(v)), 4) if v else None}
        for k, v in bands.items()
    }
    print("\n  Calibration")
    for k, v in report["calibration"].items():
        if v["n"]:
            print(f"    {k:<14} n={v['n']:<5} observed accuracy {v['accuracy'] * 100:5.1f}%")

    (ROOT / "evaluation.json").write_text(json.dumps(report, indent=2))
    draw_confusion(cm_new, names, "Dermora AI - confusion matrix", ROOT / "confusion_matrix.png")
    print(f"\n  Wrote evaluation.json and confusion_matrix.png")
    print(f"  Class distribution in split: {dict(Counter(names[t] for t in truths))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
