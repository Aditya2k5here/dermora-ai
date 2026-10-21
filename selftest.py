"""
Dermora AI - pre-flight check.

Run this before the demo. It exercises every stage of the pipeline against the
bundled sample images and tells you, in order, exactly which component fails if
one does.

    python selftest.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

OK, BAD, WARN = "  [ ok ]", "  [FAIL]", "  [warn]"
failures: list[str] = []


def step(title: str):
    print(f"\n• {title}")


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"{BAD} {msg}")


def main() -> int:
    print("=" * 66)
    print("  DERMORA AI - self test")
    print("=" * 66)

    # ---------------------------------------------------------------- deps
    step("Dependencies")
    for mod, label in [("numpy", "numpy"), ("cv2", "opencv-python"),
                       ("torch", "torch"), ("ultralytics", "ultralytics"),
                       ("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
                       ("multipart", "python-multipart"), ("reportlab", "reportlab")]:
        try:
            m = __import__(mod)
            print(f"{OK} {label:<18} {getattr(m, '__version__', '')}")
        except ImportError:
            fail(f"{label} is not installed  ->  pip install {label}")

    if failures:
        print("\nInstall the missing packages and re-run.")
        return 1

    import cv2
    import numpy as np

    # ------------------------------------------------------------- weights
    step("Checkpoint")
    weights = ROOT / "model" / "best.pt"
    if not weights.exists():
        fail(f"weights not found at {weights}")
        return 1
    print(f"{OK} {weights.name} ({weights.stat().st_size / 1e6:.1f} MB)")

    from dermora.cardinfo import read_checkpoint_meta
    meta = read_checkpoint_meta(weights)
    if meta.get("error"):
        fail(meta["error"])
    else:
        print(f"{OK} metadata: {meta['nc']} classes {meta['classes']}")
        print(f"{OK} trained {meta['trained']} - mAP50 {meta['metrics']['mAP50']:.3f}, "
              f"recall {meta['metrics']['recall']:.3f}")

    # -------------------------------------------------------------- engine
    step("Model load")
    from dermora.pipeline import Pipeline
    t0 = time.perf_counter()
    try:
        pipe = Pipeline(weights, imgsz=224, use_tta=True)
    except Exception as exc:
        fail(f"engine failed to initialise: {exc}")
        traceback.print_exc()
        return 1
    print(f"{OK} loaded in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------- samples
    step("Inference")
    samples = sorted([p for p in (ROOT / "samples").glob("*")
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not samples:
        print(f"{WARN} no sample images in ./samples - using a synthetic image")
        synthetic = np.full((450, 600, 3), 190, np.uint8)
        cv2.circle(synthetic, (300, 225), 90, (70, 50, 45), -1)
        cases = [("synthetic.jpg", cv2.cvtColor(synthetic, cv2.COLOR_BGR2RGB))]
    else:
        cases = [(p.name, cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
                 for p in samples[:4]]

    last = None
    for name, img in cases:
        try:
            t0 = time.perf_counter()
            case, images = pipe.analyse(img, filename=name)
            dt = (time.perf_counter() - t0) * 1000
            last = (case, images)
        except Exception as exc:
            fail(f"analysis of {name} failed: {exc}")
            traceback.print_exc()
            continue

        p = case["prediction"]
        total = sum(p["probs"].values())
        nonzero = sum(1 for v in p["probs"].values() if v > 1e-4)

        print(f"\n   {name}")
        print(f"   {'-' * 58}")
        for r in p["ranked"][:4]:
            bar = "█" * max(1, int(r["prob"] * 34))
            print(f"     {r['code']:<6} {r['prob'] * 100:5.1f}%  {bar}")
        print(f"     triage {case['triage']['band']:<9} concern {case['concern'] * 100:.1f}%  "
              f"quality {case['quality']['score']:.0f}  {dt:.0f} ms")

        if abs(total - 1.0) > 1e-3:
            fail(f"posterior does not sum to 1 for {name} (got {total:.4f})")
        if nonzero < 2:
            fail(f"posterior collapsed to one-hot for {name} - raw head aggregation is "
                 f"not working")
        if p.get("degraded"):
            print(f"{WARN} DEGRADED PATH: {p.get('degraded_reason')}")
            failures.append("engine fell back to the post-NMS path")
        if not case["morphology"].get("available"):
            print(f"{WARN} morphometry unavailable: {case['morphology'].get('error')}")
        if not case["attention"].get("available"):
            print(f"{WARN} attention map unavailable")
        if not images.get("attention"):
            print(f"{WARN} attention overlay image not produced")

    # ----------------------------------------------------------- reporting
    step("PDF report")
    if last:
        try:
            from dermora.report import build_report
            pdf = build_report(*last)
            out = ROOT / "selftest_report.pdf"
            out.write_bytes(pdf)
            print(f"{OK} wrote {out.name} ({len(pdf) / 1024:.0f} KB)")
        except Exception as exc:
            fail(f"PDF generation failed: {exc}")
            traceback.print_exc()

    # ------------------------------------------------------------- verdict
    print("\n" + "=" * 66)
    if failures:
        print(f"  {len(failures)} PROBLEM(S) FOUND:")
        for f in failures:
            print(f"    - {f}")
        print("=" * 66)
        return 1
    print("  ALL CHECKS PASSED - start the app with:  python app.py")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
