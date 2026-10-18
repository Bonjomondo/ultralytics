#!/usr/bin/env python3
"""Strict replication of the RT-DETR baseline recipe from the SO-DETR paper.

Reproduces the training protocol reported in:

    Zhang et al., "SO-DETR: Leveraging Dual-Domain Features and Knowledge
    Distillation for Small Object Detection".

The paper's RT-DETR baselines (Table I) are ResNet-backed RT-DETR models:
    * RT-DETR-R18  : ResNet18, 20.0M params, 57.3 GFLOPs  (AP 26.7 / AP50 44.6)
    * RT-DETR-R50  : ResNet50, 42.0M params, 129.9 GFLOPs (AP 28.4 / AP50 47.0)

This fork's `rtdetr-resnet50.yaml` reproduces the RT-DETR-R50 architecture
(42.9M params), so RT-DETR-R50 is the faithful replication target here.

Recipe (paper text + SO-DETR repo V1.0 baseline protocol):
    epochs=350, batch=4, patience=40, imgsz=640
    AdamW, lr0=1e-4, lrf=1.0 (constant LR), weight_decay=1e-4
    warmup_epochs=3, nbs=64, mixup=0.2, close_mosaic=0, seed=0, max_det=300

Fidelity caveat (pretrained weights)
------------------------------------
The paper initializes from COCO-pretrained RT-DETR weights ("default parameters
from the official implementations"). Ultralytics ships COCO checkpoints only for
the HGNetv2 variants (`rtdetr-l.pt`, `rtdetr-x.pt`); there is **no** Ultralytics
COCO checkpoint for the ResNet RT-DETR configs. Therefore, by default this script
trains RT-DETR-R50 **from scratch**. Supply official/converted weights with
`--pretrained <path/to.pt>` to match the paper's initialization.

Examples
--------
Smoke test (fast, validates the whole pipeline)::

    PYTHONPATH=$PWD python train_paper_repro_visdrone.py \
        --epochs 1 --fraction 0.03 --name smoke

Full strict replication (RT-DETR-R50, paper recipe)::

    PYTHONPATH=$PWD python train_paper_repro_visdrone.py --model r50
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import RTDETR

ROOT = Path(__file__).resolve().parent

MODEL_CONFIGS = {
    "r50": ROOT / "ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml",
    "r101": ROOT / "ultralytics/cfg/models/rt-detr/rtdetr-resnet101.yaml",
    "l": ROOT / "ultralytics/cfg/models/rt-detr/rtdetr-l.yaml",
}

DEFAULT_DATA = ROOT / "configs/visdrone-local.yaml"

# Strict paper recipe (see module docstring for provenance).
PAPER_RECIPE = dict(
    epochs=350,
    patience=40,
    batch=4,
    imgsz=640,
    optimizer="AdamW",
    lr0=0.0001,
    lrf=1.0,
    weight_decay=0.0001,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    nbs=64,
    mixup=0.2,
    close_mosaic=0,
    amp=True,
    deterministic=False,  # F.grid_sample in RT-DETR does not support deterministic ops
    seed=0,
    max_det=300,
    val=True,
    plots=True,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_CONFIGS),
        default="r50",
        help="RT-DETR variant: r50 (paper Table I default), r101, or l (HGNetv2).",
    )
    parser.add_argument("--model-cfg", default=None, help="Optional custom model YAML (overrides --model).")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset YAML.")
    parser.add_argument("--pretrained", default=None, help="Optional pretrained .pt to initialize from.")
    parser.add_argument("--device", default="0", help="CUDA device(s), e.g. 0 or 0,1.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of the training set to use.")
    parser.add_argument("--project", default="paper_repro", help="Run project dir (relative to runs_dir unless absolute).")
    parser.add_argument("--name", default=None)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--resume", action="store_true")

    # Allow overriding the strict recipe knobs when needed (defaults = paper).
    for key, value in PAPER_RECIPE.items():
        parser.add_argument(f"--{key.replace('_', '-')}", dest=key, type=type(value), default=value)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_cfg = Path(args.model_cfg) if args.model_cfg else MODEL_CONFIGS[args.model]
    if not model_cfg.exists():
        raise FileNotFoundError(f"Model config not found: {model_cfg}")

    name = args.name or f"rtdetr-{args.model}-paper-350e-s{args.seed}"
    model_source = args.pretrained if args.pretrained else str(model_cfg)

    print("=" * 72)
    print("Strict paper-recipe replication (SO-DETR paper RT-DETR baseline)")
    print(f"  model cfg  : {model_cfg}")
    print(f"  initialize : {'pretrained ' + str(args.pretrained) if args.pretrained else 'from scratch (no Ultralytics COCO weights for ResNet RT-DETR)'}")
    print(f"  data       : {args.data}")
    print(f"  recipe     : {PAPER_RECIPE}")
    print("=" * 72)

    model = RTDETR(model_source)

    train_args = {k: getattr(args, k) for k in PAPER_RECIPE}
    train_args.update(
        data=args.data,
        device=args.device,
        workers=args.workers,
        fraction=args.fraction,
        project=args.project,
        name=name,
        exist_ok=args.exist_ok,
        resume=args.resume,
    )
    model.train(**train_args)


if __name__ == "__main__":
    main()