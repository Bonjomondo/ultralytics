"""Matched seed-0 continuation from stripped 150-epoch EMA weights (optimizer restart)."""
import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import torch

from experiments.r18_p2_sfi.evaluate import digest, write
from experiments.r18_p2_sfi.train import MODELS, RECIPE, SFIHealth, SFITrainer, tensor_digest
from ultralytics import RTDETR

ROOT = Path(__file__).resolve().parents[2]
KEY = "metrics/mAP50-95(B)"


def plateau_status(values, minimum=150, patience=50, window=20, tolerance=0.001):
    """Require both a best-score plateau and a plateau of consecutive window means."""
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("Nonfinite convergence metric")
    enough = len(values) >= max(minimum, patience + 1, 2 * window)
    if not enough:
        return {"converged": False, "epochs": len(values), "reason": "minimum observation budget not reached"}
    best_gain = max(values[-patience:]) - max(values[:-patience])
    mean_gain = sum(values[-window:]) / window - sum(values[-2 * window:-window]) / window
    return {"converged": best_gain <= tolerance and abs(mean_gain) <= tolerance,
            "epochs": len(values), "best_gain_last50": best_gain,
            "mean_gain_last20_vs_previous20": mean_gain, "tolerance": tolerance,
            "criterion": "after >=150 additional epochs; <=0.1pp best gain over 50 epochs and abs(mean20-previous20)<=0.1pp"}


class ContinuationTrainer(SFITrainer):
    """Keep an unstripped restart checkpoint even after normal final evaluation."""

    def final_eval(self):
        if self.last.exists():
            shutil.copy2(self.last, self.wdir / "resume.pt")
        super().final_eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("c0", "c2"), required=True)
    parser.add_argument("--additional-epochs", type=int, default=350)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/detect/r18_p2_sfi_convergence_20261024")
    parser.add_argument("--name")
    parser.add_argument("--fraction", type=float, default=1.0, help="Smoke tests only if less than 1")
    parser.add_argument("--minimum-epochs", type=int, default=150)
    parser.add_argument("--resume", type=Path, help="Resume an interrupted continuation from its unstripped last/resume.pt")
    args = parser.parse_args()
    source_run = ROOT / f"runs/detect/r18_p2_sfi_20261023/rtdetr-r18-{args.variant}-150e-s0"
    source = args.resume.resolve() if args.resume else source_run / "weights/last.pt"
    rows = list(csv.DictReader((source_run / "results.csv").open()))
    if len(rows) != 150 or int(rows[-1]["epoch"]) != 150:
        raise RuntimeError("Continuation requires a completed 150-epoch source")
    model = RTDETR(str(source))
    expected = RTDETR(str(MODELS[args.variant])).model.yaml
    if any(model.model.yaml[key] != expected[key] for key in ("backbone", "head")):
        raise ValueError("Checkpoint architecture mismatch")
    initial_hash = tensor_digest(model.model.state_dict())
    source_hash = digest(source)
    SFIHealth().attach(model)

    def start(trainer):
        actual_hash = tensor_digest(trainer.model.state_dict())
        if actual_hash != initial_hash:
            raise RuntimeError("Trainer changed the continuation initialization")
        write(trainer.save_dir / ("resume_verification.json" if args.resume else "continuation.json"), {
            "variant": args.variant, "seed": 0, "source_checkpoint": str(source),
            "source_checkpoint_sha256": source_hash, "source_completed_epochs": 150,
            "mode": ("resume within continuation stage with saved optimizer and EMA counters" if args.resume else
                     "warm start from stripped FP16 EMA; fresh AdamW and EMA counters; not exact resume"),
            "loaded_state_sha256": actual_hash, "loaded_state_verified": True,
            "maximum_additional_epochs": args.additional_epochs, "minimum_additional_epochs": args.minimum_epochs,
            "postprocess": "sorted confidence mask fixed; conf=0.001",
            "lr": "same constant 1e-4 with 3-epoch re-warmup because optimizer is fresh",
        })

    def saved(trainer):
        completed = trainer.epoch + 1
        if completed % 50 == 0:
            shutil.copy2(trainer.last, trainer.wdir / f"additional{completed}_last.pt")
            shutil.copy2(trainer.best, trainer.wdir / f"additional{completed}_best.pt")

    def epoch_end(trainer):
        rows = list(csv.DictReader(trainer.csv.open()))
        status = plateau_status([float(row[KEY]) for row in rows], minimum=args.minimum_epochs)
        status.update(cumulative_epochs=150 + len(rows), additional_epochs=len(rows),
                      maximum_additional_epochs=args.additional_epochs)
        if len(rows) >= args.additional_epochs and not status["converged"]:
            status["reason"] = "budget cap reached; convergence not established"
        write(trainer.save_dir / "convergence_status.json", status)
        if status["converged"]:
            trainer.stop = True

    model.add_callback("on_train_start", start)
    model.add_callback("on_model_save", saved)
    model.add_callback("on_fit_epoch_end", epoch_end)
    torch.set_num_threads(4)
    recipe = {**RECIPE, "epochs": args.additional_epochs, "patience": 0, "conf": 0.001, "save_period": -1}
    model.train(trainer=ContinuationTrainer, **recipe, data=str(ROOT / "configs/visdrone-local.yaml"),
                device="0", workers=0, fraction=args.fraction, project=str(args.project.resolve()),
                name=args.name or f"rtdetr-r18-{args.variant}-seed0-cont", pretrained=False,
                exist_ok=bool(args.resume), resume=str(source) if args.resume else False)


if __name__ == "__main__":
    main()
