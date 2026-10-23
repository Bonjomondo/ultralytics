#!/usr/bin/env python3
"""Run R0/R1 or P5-NAIFI R2 ablations on VisDrone, with training health logs.

    python train_rtdetr_r18_na.py --variant r1 --device 0
    python train_rtdetr_r18_na.py --variant r0 --device 0

Defaults: 150 epochs, batch 8, AdamW, FP32, zero bias warmup LR, no early stopping.
Both variants receive exactly the same seeded initialization for all shared weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from train_paper_repro_visdrone import DEFAULT_DATA, PAPER_RECIPE, parse_bool
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import ResidualLocalRefine
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "ultralytics/cfg/models/rt-detr"
MODEL_CONFIGS = {
    "r0": CONFIG_DIR / "rtdetr-resnet18.yaml",
    "r1": CONFIG_DIR / "rtdetr-resnet18-p4-na5.yaml",
    "r2": CONFIG_DIR / "rtdetr-resnet18-p5-naifi.yaml",
    "r2a": CONFIG_DIR / "rtdetr-resnet18-p5-naifi-no-rpb.yaml",
    "r2b": CONFIG_DIR / "rtdetr-resnet18-p5-naifi.yaml",
    "r2c": CONFIG_DIR / "rtdetr-resnet18-p5-naifi-k11.yaml",
}
RECIPE = {
    **PAPER_RECIPE,
    "epochs": 150,
    "batch": 8,
    "patience": 0,
    "amp": False,
    "warmup_bias_lr": 0.0,
    "momentum": 0.9,
    "cache": False,
}


def build_ablation_model(cfg, nc=10, ch=3, seed=0, weights=None, verbose=False):
    """Initialize shared parameters from R0, independent of NA's consumption of random numbers."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        baseline = RTDETRDetectionModel(str(MODEL_CONFIGS["r0"]), nc=nc, ch=ch, verbose=False)
        model = RTDETRDetectionModel(cfg, nc=nc, ch=ch, verbose=verbose)
        if weights is not None:
            if weights.yaml["backbone"] != baseline.yaml["backbone"]:
                raise ValueError("Initialization checkpoint must use this experiment's ResNet18-vd backbone")
            baseline.load(weights, verbose=False)
        reference = baseline.state_dict()
        initial = model.state_dict()
        common = {name: value for name, value in reference.items()
                  if name in initial and initial[name].shape == value.shape}
        model.load_state_dict(common, strict=False)
        if weights is not None:
            model.load(weights, verbose=verbose)
        state = model.state_dict()
        digest = hashlib.sha256()
        reference_digest = hashlib.sha256()
        baseline_digest = hashlib.sha256()
        for name in sorted(reference):
            baseline_digest.update(name.encode())
            baseline_digest.update(reference[name].detach().cpu().contiguous().numpy().tobytes())
        for name in sorted(common):
            if not torch.equal(state[name], reference[name]):
                raise RuntimeError(f"Shared initialization mismatch: {name}")
            digest.update(name.encode())
            digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
            reference_digest.update(name.encode())
            reference_digest.update(reference[name].detach().cpu().contiguous().numpy().tobytes())
        model.ablation_initialization = {
            "seed": seed,
            "shared_state_sha256": digest.hexdigest(),
            "reference_shared_state_sha256": reference_digest.hexdigest(),
            "r0_full_state_sha256": baseline_digest.hexdigest(),
            "shared_state_items": len(common),
            "shared_state_verified": True,
            "removed_r0_state": sorted(set(reference) - set(common)),
            "new_variant_state": sorted(set(state) - set(common)),
            "shape_mismatches": {name: [list(reference[name].shape), list(state[name].shape)]
                                 for name in set(reference) & set(state)
                                 if reference[name].shape != state[name].shape},
            "source": "checkpoint" if weights is not None else "seeded R0 from scratch",
        }
    return model


class R18AblationTrainer(RTDETRTrainer):
    """Apply paired initialization to the actual training model, including trainer rebuilds."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Keep the selected YAML architecture when transferring checkpoint weights."""
        return build_ablation_model(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            seed=self.args.seed,
            weights=weights,
            verbose=verbose,
        )


class TrainingHealth:
    """Sample the first three training batches per epoch and write health.jsonl."""

    def __init__(self):
        """Keep hooks and detached scalar samples only."""
        self.handles = []
        self.capture = False
        self.samples = {}
        self.written_epochs = set()
        self.epoch_times = []

    def _sample(self, name, value):
        """Store a detached scalar without retaining an autograd graph."""
        if self.capture:
            self.samples.setdefault(name, []).append(value.detach())

    def start(self, trainer):
        """Attach hooks to the training model; EMA validation is not sampled."""
        model = unwrap_model(trainer.model)
        self.refine = next((m for m in model.modules() if isinstance(m, ResidualLocalRefine)), None)
        head = model.model[-1]
        last_bbox = head.dec_bbox_head[-1]

        def hidden_hook(module, inputs, output):
            if module.training:
                self._sample("last_bbox_relu_active_ratio", (output > 0).float().mean())

        def delta_hook(module, inputs, output):
            if module.training:
                self._sample("last_bbox_delta_logit_abs", output.float().abs().mean())

        def box_hook(module, inputs, output):
            if module.training:
                boxes = output[0]
                self._sample("last_bbox_step_abs", (boxes[-1] - boxes[-2]).float().abs().mean())

        self.handles.extend(
            (
                last_bbox.layers[-2].register_forward_hook(hidden_hook),
                last_bbox.register_forward_hook(delta_hook),
                head.register_forward_hook(box_hook),
            )
        )
        if self.refine is not None:
            self.handles.append(self.refine.gamma.register_hook(lambda grad: self._sample("gamma_grad_abs", grad.abs())))
            self.handles.append(
                self.refine.attn.qkv.weight.register_hook(
                    lambda grad: self._sample("na_qkv_grad_abs", grad.abs().mean())
                )
            )
        manifest = {
            **model.ablation_initialization,
            "model_config": str(trainer.args.model),
            "amp": trainer.amp,
            "gradient_sampling": "first three training batches per epoch; scaled if AMP is enabled",
        }
        (trainer.save_dir / "initialization.json").write_text(json.dumps(manifest, indent=2) + "\n")
        LOGGER.info(f"Paired shared-state SHA256: {manifest['shared_state_sha256']}")

    def epoch_start(self, trainer):
        """Measure training time separately from validation and checkpoint writing."""
        if trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)
            torch.cuda.reset_peak_memory_stats(trainer.device)
        self.started = time.perf_counter()
        self.batches = 0
        self.samples = {}

    def batch_start(self, trainer):
        """Limit diagnostic work to three batches per epoch."""
        self.capture = self.batches < 3

    def batch_end(self, trainer):
        """Stop sampling until the next training batch."""
        self.capture = False
        self.batches += 1

    def epoch_end(self, trainer):
        """Snapshot training cost before validation starts."""
        if trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)
        self.elapsed = time.perf_counter() - self.started
        self.peak_mib = (
            torch.cuda.max_memory_allocated(trainer.device) / 2**20 if trainer.device.type == "cuda" else None
        )

    def fit_epoch_end(self, trainer):
        """Persist diagnostics once per epoch, including a first-five-epochs mean."""
        epoch = trainer.epoch + 1
        if epoch in self.written_epochs:
            return
        self.written_epochs.add(epoch)
        self.epoch_times.append(self.elapsed)
        record = {"epoch": epoch, "train_seconds": self.elapsed, "train_peak_allocated_mib": self.peak_mib}
        for name, values in self.samples.items():
            value = torch.stack(values).mean().item()
            record[name] = value if math.isfinite(value) else None
        record["gamma"] = self.refine.gamma.detach().item() if self.refine is not None else None
        if len(self.epoch_times) == 5:
            record["mean_train_seconds_first5"] = sum(self.epoch_times) / 5
        with (trainer.save_dir / "health.jsonl").open("a") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")
        LOGGER.info(f"R18 health: {record}")

    def close(self, trainer):
        """Remove hooks when training completes."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def attach(self, model):
        """Register callbacks on the RTDETR wrapper."""
        for event, callback in {
            "on_train_start": self.start,
            "on_train_epoch_start": self.epoch_start,
            "on_train_batch_start": self.batch_start,
            "on_train_batch_end": self.batch_end,
            "on_train_epoch_end": self.epoch_end,
            "on_fit_epoch_end": self.fit_epoch_end,
            "on_train_end": self.close,
        }.items():
            model.add_callback(event, callback)


def parse_args(argv=None):
    """Parse a standalone R18 recipe, including explicit boolean values."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=MODEL_CONFIGS, default="r1")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--device", default="0", help="One CUDA device, cpu, or mps.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--project", default="r18_na_ablation")
    parser.add_argument("--name", default=None)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--pretrained", type=Path, default=None, help="Compatible local R18 .pt checkpoint.")
    initialization.add_argument("--resume", type=Path, default=None, help="Resume the selected variant's last.pt.")
    for name, value in RECIPE.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            type=parse_bool if isinstance(value, bool) else type(value),
            default=value,
        )
    args = parser.parse_args(argv)
    if "," in args.device:
        parser.error("This diagnostic runner supports one device per experiment")
    for checkpoint in (args.pretrained, args.resume):
        if checkpoint is not None and not checkpoint.is_file():
            parser.error(f"Checkpoint does not exist: {checkpoint}")
    return args


def main():
    """Start the selected experiment and record decoder/gate health throughout training."""
    args = parse_args()
    cfg = MODEL_CONFIGS[args.variant]
    model = RTDETR(str(args.resume or cfg))
    if args.resume:
        expected = RTDETR(str(cfg)).model.yaml
        actual = model.model.yaml
        if any(actual[key] != expected[key] for key in ("backbone", "head")):
            raise ValueError("Resume checkpoint does not match the selected R18 architecture/window/RPB")
    elif args.pretrained:
        model.load(str(args.pretrained))
    TrainingHealth().attach(model)
    recipe = {name: getattr(args, name) for name in RECIPE}
    LOGGER.info(f"R18 {args.variant}: {cfg}\nEffective recipe: {recipe}")
    model.train(
        trainer=R18AblationTrainer,
        **recipe,
        data=args.data,
        device=args.device,
        workers=args.workers,
        fraction=args.fraction,
        project=args.project,
        name=args.name or f"rtdetr-r18-{args.variant}-{args.epochs}e-s{args.seed}",
        pretrained=str(args.pretrained) if args.pretrained else False,
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
