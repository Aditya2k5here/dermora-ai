"""
Dermora AI - application server.

    python app.py                 # http://127.0.0.1:8000
    python app.py --port 9000 --no-tta

The model is loaded on a background thread so the interface is interactive
immediately and reports its own readiness, rather than blocking the first
request behind a cold start in front of an audience.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from dermora import __version__
from dermora.cardinfo import read_checkpoint_meta
from dermora.knowledge import LESIONS, RISK_TIERS
from dermora.pipeline import Pipeline

ROOT = Path(__file__).parent.resolve()
STATIC = ROOT / "static"
SAMPLES = ROOT / "samples"
WEIGHTS = Path(os.environ.get("DERMORA_WEIGHTS", ROOT / "model" / "best.pt"))
IMGSZ = int(os.environ.get("DERMORA_IMGSZ", 224))
USE_TTA = os.environ.get("DERMORA_TTA", "1") != "0"
MAX_UPLOAD_MB = 25
MAX_CASES = 60

ALLOWED_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

app = FastAPI(title="Dermora AI", version=__version__,
              description="Dermoscopic lesion analysis and triage support")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# ---------------------------------------------------------------------------
# model lifecycle
# ---------------------------------------------------------------------------

_state = {"status": "loading", "error": None, "loaded_at": None, "load_seconds": None}
_pipeline: Pipeline | None = None
_cases: "OrderedDict[str, tuple[dict, dict]]" = OrderedDict()
_cases_lock = threading.Lock()


def _load_model() -> None:
    global _pipeline
    t0 = time.perf_counter()
    try:
        if not WEIGHTS.exists():
            raise FileNotFoundError(
                f"Weights not found at {WEIGHTS}. Place best.pt in ./model/ or set "
                f"DERMORA_WEIGHTS."
            )
        _pipeline = Pipeline(WEIGHTS, imgsz=IMGSZ, use_tta=USE_TTA)
        _state.update(status="ready", loaded_at=time.time(),
                      load_seconds=round(time.perf_counter() - t0, 2))
        print(f"[dermora] model ready in {_state['load_seconds']}s "
              f"({'TTA on' if USE_TTA else 'TTA off'}, imgsz={IMGSZ})")
    except Exception as exc:
        _state.update(status="error", error=str(exc))
        print(f"[dermora] MODEL LOAD FAILED: {exc}", file=sys.stderr)


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_load_model, daemon=True).start()


def _require_pipeline() -> Pipeline:
    if _state["status"] == "error":
        raise HTTPException(503, f"Model unavailable: {_state['error']}")
    if _pipeline is None:
        raise HTTPException(503, "Model is still loading - retry in a moment.")
    return _pipeline


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _decode_image(raw: bytes, name: str) -> np.ndarray:
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_MB} MB limit.")
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"'{name}' could not be decoded as an image.")
    h, w = img.shape[:2]
    if min(h, w) < 48:
        raise HTTPException(400, "Image is too small to analyse (minimum 48px).")
    if max(h, w) > 4200:                       # guard against enormous phone shots
        scale = 4200 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _store(case: dict, images: dict) -> None:
    with _cases_lock:
        _cases[case["case_id"]] = (case, images)
        while len(_cases) > MAX_CASES:
            _cases.popitem(last=False)


def _summary(case: dict) -> dict:
    return {
        "case_id": case["case_id"],
        "filename": case["filename"],
        "timestamp": case["timestamp"],
        "top_class": case["prediction"]["top_class"],
        "top_name": case["prediction"]["top_name"],
        "confidence": case["prediction"]["confidence"],
        "concern": case["concern"],
        "band": case["triage"]["band"],
        "risk": case["triage"]["risk"],
        "quality": case["quality"]["score"],
        "tds": case["morphology"].get("abcd", {}).get("tds") if case["morphology"].get("available") else None,
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {
        "service": "Dermora AI",
        "version": __version__,
        "model_status": _state["status"],
        "error": _state["error"],
        "load_seconds": _state["load_seconds"],
        "tta": USE_TTA,
        "imgsz": IMGSZ,
        "cases_in_session": len(_cases),
    }


@app.get("/api/model-card")
def model_card() -> dict:
    meta = read_checkpoint_meta(WEIGHTS)
    meta["runtime"] = {
        "status": _state["status"],
        "tta_views": 4 if USE_TTA else 1,
        "inference_size": IMGSZ,
        "device": "cpu",
    }
    meta["taxonomy"] = [
        {
            "code": code,
            "name": rec["name"],
            "full_name": rec["full_name"],
            "category": rec["category"],
            "risk": rec["risk"],
            "color": RISK_TIERS[rec["risk"]]["color"],
            "prevalence": rec["prevalence"],
            "summary": rec["summary"],
            "dermoscopy": rec["dermoscopy"],
            "management": rec["management"],
            "urgency": rec["urgency"],
        }
        for code, rec in LESIONS.items()
    ]
    meta["limitations"] = [
        "Trained on ISIC 2018 Task 3 (HAM10000): 10,015 images, heavily skewed towards "
        "Fitzpatrick I-III skin. Performance on darker skin is not characterised.",
        "Class imbalance is severe - 66.9% of training images are melanocytic nevi, "
        "against 1.1% dermatofibroma. Rare-class predictions carry wider error bars.",
        "Ground-truth boxes span the whole frame, so the model localises nothing; the "
        "attention map and the segmentation contour come from separate methods.",
        "Validation metrics are detection metrics (mAP) computed against whole-image "
        "boxes, which is why mAP is high while precision sits at 0.59.",
        "The dataset contains dermoscopic images. Ordinary phone photographs are out of "
        "distribution and will degrade accuracy.",
    ]
    return meta


@app.get("/api/samples")
def samples() -> dict:
    if not SAMPLES.exists():
        return {"samples": []}
    items = []
    for p in sorted(SAMPLES.iterdir()):
        if p.suffix.lower() in ALLOWED_SUFFIX:
            items.append({"name": p.name, "url": f"/api/samples/{p.name}"})
    return {"samples": items}


@app.get("/api/samples/{name}")
def sample_file(name: str) -> FileResponse:
    path = (SAMPLES / name).resolve()
    if not str(path).startswith(str(SAMPLES.resolve())) or not path.exists():
        raise HTTPException(404, "Sample not found")
    return FileResponse(path)


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...),
                  morphology: bool = Query(True, description="Run classical morphometry")) -> JSONResponse:
    pipeline = _require_pipeline()
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix and suffix not in ALLOWED_SUFFIX:
        raise HTTPException(400, f"Unsupported file type '{suffix}'.")

    img = _decode_image(await file.read(), file.filename or "upload")
    case, images = pipeline.analyse(img, filename=file.filename or "upload",
                                    with_morphology=morphology)
    _store(case, images)
    return JSONResponse(pipeline.as_payload(case, images))


@app.post("/api/analyze-sample")
def analyze_sample(name: str = Query(...)) -> JSONResponse:
    pipeline = _require_pipeline()
    path = (SAMPLES / name).resolve()
    if not str(path).startswith(str(SAMPLES.resolve())) or not path.exists():
        raise HTTPException(404, "Sample not found")
    img = _decode_image(path.read_bytes(), name)
    case, images = pipeline.analyse(img, filename=name)
    _store(case, images)
    return JSONResponse(pipeline.as_payload(case, images))


@app.post("/api/batch")
async def batch(files: list[UploadFile] = File(...)) -> JSONResponse:
    """Triage mode: many images in, one sortable table out."""
    pipeline = _require_pipeline()
    if len(files) > 40:
        raise HTTPException(400, "Batch limit is 40 images.")

    results, errors = [], []
    for f in files:
        try:
            img = _decode_image(await f.read(), f.filename or "upload")
            case, images = pipeline.analyse(img, filename=f.filename or "upload")
            _store(case, images)
            summary = _summary(case)
            summary["thumb"] = pipeline.as_payload({}, {"thumb": images["thumb"]})["images"]["thumb"]
            summary["caveats"] = len(case["caveats"])
            results.append(summary)
        except HTTPException as exc:
            errors.append({"filename": f.filename, "error": exc.detail})
        except Exception as exc:                                  # pragma: no cover
            errors.append({"filename": f.filename, "error": str(exc)})

    rank = {"URGENT": 0, "PRIORITY": 1, "REVIEW": 2, "ROUTINE": 3}
    results.sort(key=lambda r: (rank.get(r["band"], 9), -r["concern"]))
    return JSONResponse({"results": results, "errors": errors, "count": len(results)})


@app.get("/api/cases")
def list_cases() -> dict:
    import base64
    with _cases_lock:
        entries = list(_cases.values())
    items = []
    for case, images in reversed(entries):
        row = _summary(case)
        thumb = images.get("thumb")
        if thumb:
            row["thumb"] = "data:image/jpeg;base64," + base64.b64encode(thumb).decode("ascii")
        items.append(row)
    return {"cases": items, "count": len(items)}


@app.get("/api/case/{case_id}")
def get_case(case_id: str) -> JSONResponse:
    entry = _cases.get(case_id)
    if not entry:
        raise HTTPException(404, "Case not found in this session.")
    case, images = entry
    return JSONResponse(_require_pipeline().as_payload(case, images))


@app.get("/api/case/{case_id}/report.pdf")
def case_report(case_id: str) -> Response:
    entry = _cases.get(case_id)
    if not entry:
        raise HTTPException(404, "Case not found in this session.")
    case, images = entry
    try:
        from dermora.report import build_report
        pdf = build_report(case, images)
    except ImportError:
        raise HTTPException(500, "ReportLab is not installed - run: pip install reportlab")
    return Response(
        pdf, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="Dermora_{case_id}.pdf"'},
    )


@app.delete("/api/cases")
def clear_cases() -> dict:
    with _cases_lock:
        n = len(_cases)
        _cases.clear()
    return {"cleared": n}


# ---------------------------------------------------------------------------
# static front end
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main() -> None:
    global IMGSZ, USE_TTA, WEIGHTS
    ap = argparse.ArgumentParser(description="Dermora AI server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--no-tta", action="store_true", help="single view instead of 4-view TTA")
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    WEIGHTS = Path(args.weights)
    IMGSZ = args.imgsz
    USE_TTA = not args.no_tta

    banner = f"""
    ┌────────────────────────────────────────────────┐
    │  DERMORA AI  v{__version__}                              │
    │  Dermoscopic lesion analysis & triage support  │
    ├────────────────────────────────────────────────┤
    │  Interface : http://{args.host}:{args.port}{' ' * (17 - len(str(args.port)) - len(args.host))}│
    │  API docs  : http://{args.host}:{args.port}/docs{' ' * (12 - len(str(args.port)) - len(args.host))}│
    │  Weights   : {WEIGHTS.name}{' ' * max(0, 34 - len(WEIGHTS.name))}│
    └────────────────────────────────────────────────┘
    """
    print(banner)
    uvicorn.run("app:app" if args.reload else app, host=args.host, port=args.port,
                reload=args.reload, log_level="warning")


if __name__ == "__main__":
    main()
