"""
Dermora AI - checkpoint introspection.

Reads the provenance stored inside best.pt: training date, Ultralytics version,
hyper-parameters, final validation metrics and the full per-epoch training
history.

Deliberately implemented with `zipfile` plus a stubbed `pickle.Unpickler`
rather than `torch.load`. A .pt file is a zip archive containing a pickle, and
the metadata lives in plain Python containers, so no tensors need to be
materialised. That means the model card renders instantly, works without torch
installed, and never executes arbitrary classes from the checkpoint.
"""

from __future__ import annotations

import io
import pickle
import zipfile
from pathlib import Path
from typing import Any

_PRIMITIVES = (str, int, float, bool, type(None))


class _Stub:
    """Placeholder for any class referenced by the pickle."""

    def __init__(self, *_a, **_k):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state

    def __reduce__(self):
        return (_Stub, ())


class _SafeUnpickler(pickle.Unpickler):
    """Resolves every foreign class to an inert stub; loads no tensor storage."""

    def find_class(self, module: str, name: str):
        if module == "collections" and name == "OrderedDict":
            return dict
        if module == "torch" and name == "Size":
            return tuple
        if module == "__builtin__" and name == "set":
            return set
        if module == "builtins" and name in {"set", "frozenset", "dict", "list", "tuple"}:
            return getattr(__builtins__, name) if hasattr(__builtins__, name) else set
        return type(name, (_Stub,), {})

    def persistent_load(self, _pid):
        return None


def _clean(value: Any, depth: int = 0):
    """Reduce the unpickled tree to JSON-serialisable primitives."""
    if depth > 4:
        return None
    if isinstance(value, _PRIMITIVES):
        return value
    if isinstance(value, dict):
        return {str(k): _clean(v, depth + 1) for k, v in value.items()
                if isinstance(k, _PRIMITIVES)}
    if isinstance(value, (list, tuple, set)):
        return [_clean(v, depth + 1) for v in list(value)[:200]]
    return None


def read_checkpoint_meta(path: str | Path) -> dict:
    """Extract the human-meaningful parts of an Ultralytics checkpoint."""
    path = Path(path)
    info: dict[str, Any] = {
        "file": path.name,
        "size_mb": round(path.stat().st_size / 1e6, 2) if path.exists() else None,
    }
    try:
        with zipfile.ZipFile(path) as z:
            entry = next(n for n in z.namelist() if n.endswith("data.pkl"))
            raw = z.read(entry)
        obj = _SafeUnpickler(io.BytesIO(raw)).load()
    except Exception as exc:                                     # pragma: no cover
        info["error"] = f"Could not parse checkpoint metadata: {exc}"
        return info

    if not isinstance(obj, dict):
        info["error"] = "Unexpected checkpoint layout"
        return info

    info["trained"] = _clean(obj.get("date"))
    info["ultralytics_version"] = _clean(obj.get("version"))
    info["license"] = _clean(obj.get("license"))

    args = _clean(obj.get("train_args")) or {}
    info["hyperparameters"] = {
        k: args.get(k) for k in
        ("task", "model", "data", "epochs", "imgsz", "batch", "optimizer", "lr0",
         "lrf", "momentum", "weight_decay", "warmup_epochs", "device", "pretrained",
         "close_mosaic", "mosaic", "fliplr", "hsv_h", "hsv_s", "hsv_v", "erasing",
         "auto_augment", "box", "cls", "dfl")
        if k in args
    }

    metrics = _clean(obj.get("train_metrics")) or {}
    info["metrics"] = {
        "mAP50": metrics.get("metrics/mAP50(B)"),
        "mAP50_95": metrics.get("metrics/mAP50-95(B)"),
        "precision": metrics.get("metrics/precision(B)"),
        "recall": metrics.get("metrics/recall(B)"),
        "fitness": metrics.get("fitness"),
        "val_cls_loss": metrics.get("val/cls_loss"),
        "val_box_loss": metrics.get("val/box_loss"),
    }
    if info["metrics"].get("precision") and info["metrics"].get("recall"):
        p, r = info["metrics"]["precision"], info["metrics"]["recall"]
        info["metrics"]["f1"] = round(2 * p * r / (p + r), 4) if (p + r) else None

    history = _clean(obj.get("train_results")) or {}
    curve = {}
    for key, label in (("epoch", "epoch"),
                       ("train/cls_loss", "train_cls_loss"),
                       ("val/cls_loss", "val_cls_loss"),
                       ("train/box_loss", "train_box_loss"),
                       ("val/box_loss", "val_box_loss"),
                       ("metrics/mAP50(B)", "mAP50"),
                       ("metrics/mAP50-95(B)", "mAP50_95"),
                       ("metrics/precision(B)", "precision"),
                       ("metrics/recall(B)", "recall")):
        series = history.get(key)
        if isinstance(series, list) and series:
            curve[label] = [round(float(v), 5) for v in series
                            if isinstance(v, (int, float))]
    info["history"] = curve
    info["epochs_completed"] = len(curve.get("epoch", [])) or info["hyperparameters"].get("epochs")

    # Class names live on the stubbed model object's captured state.
    for key in ("ema", "model"):
        node = obj.get(key)
        state = getattr(node, "_state", None) if node is not None else None
        if isinstance(state, dict) and isinstance(state.get("names"), dict):
            names = state["names"]
            info["classes"] = [names[i] for i in sorted(names)]
            info["nc"] = state.get("nc") or len(info["classes"])
            break

    return info
