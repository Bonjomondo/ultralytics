"""Shared identity, immutable protocol and atomic evidence writes."""
import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(os.environ.get("C2_GATED_REPO_ROOT", ROOT))
EXP = ROOT / "experiments/c2_gated_20261103"
REPORT = REPO / "docs/experiment_reports/2026-11-03_c2_gated_fusion"
RUNS = REPO / "runs/detect/c2_gated_20261103"
ARMS = ("c2", "scalar", "spatial", "fusion")
CFG = ROOT / "ultralytics/cfg/models/rt-detr"
CONFIGS = {"c2": CFG / "rtdetr-resnet18-p2-sfi-c2.yaml", **{
    arm: CFG / f"rtdetr-resnet18-c2-{arm}.yaml" for arm in ARMS[1:]}}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n")
    temp.replace(path)


def atomic_save(path, obj):
    """Serialize on CPU with stdlib pickle, reload before replacing a durable checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        torch.save(obj, f, pickle_module=pickle)
        f.flush()
        os.fsync(f.fileno())
    check = torch.load(temp, map_location="cpu", pickle_module=pickle)
    assert check["epoch"] == obj["epoch"] and check["optimizer"] is not None
    assert all(torch.isfinite(v).all() for v in check["ema"].state_dict().values())
    del check
    temp.replace(path)
