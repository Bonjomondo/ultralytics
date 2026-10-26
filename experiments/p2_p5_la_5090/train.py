"""Train one factorial arm from scratch on CUDA:0/5090. No automatic launch on import."""
import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from experiments.decoder_budget_5090.train_5090 import (
    MAP_KEY, finite_numbers, git, now, read_health, read_rows, runtime_environment, sha256_file, write_atomic,
)
from experiments.p2_p5_la_5090.model import (
    MODELS, ROOT, FactorialHealth, FactorialTrainer, inspect_model,
)
from experiments.r18_p2_sfi.train import RECIPE
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import unwrap_model

NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]


def source_hashes():
    paths = set((ROOT / "ultralytics").rglob("*.py")) | set(MODELS.values())
    for directory in ("experiments/p2_p5_la_5090", "experiments/r18_p2_sfi", "experiments/decoder_budget_5090"):
        paths.update((ROOT / directory).glob("*.py"))
    paths.update(ROOT / name for name in ("train_paper_repro_visdrone.py", "train_rtdetr_r18_na.py"))
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(paths)}


def configure():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This experiment requires an actual 5090 on CUDA:0")


def validate_args(args):
    if args.epochs < 1 or args.seed < 0 or not 0 < args.fraction <= 1:
        raise ValueError("Require epochs>=1, seed>=0 and fraction in (0,1]")
    if Path(args.name).name != args.name or not args.name.startswith("5090-"):
        raise ValueError("Name must be one directory beginning 5090-")
    args.data, args.project = args.data.resolve(), args.project.resolve()
    data = yaml.safe_load(args.data.read_text())
    names = data["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(10)]
    if names != NAMES or data.get("nc", 10) != 10 or data.get("channels", 3) != 3:
        raise ValueError("Require the established VisDrone 10-class ordering and RGB input")


def train(args):
    validate_args(args)
    configure()
    run = args.project / args.name
    run.mkdir(parents=True, exist_ok=False)
    state = {"schema": "5090.p2-p5-la.training.v1", "status": "running", "variant": args.variant,
             "seed": args.seed, "requested_epochs": args.epochs, "fraction": args.fraction,
             "role": "fixed_budget_screen" if args.epochs >= 150 and args.fraction == 1 else "engineering_smoke",
             "started_utc": now(), "source_commit": git("rev-parse", "HEAD"),
             "source_branch": git("branch", "--show-current"), "source_dirty_paths": git("status", "--porcelain").splitlines(),
             "source_sha256": source_hashes(), "data_yaml": str(args.data), "data_yaml_sha256": sha256_file(args.data),
             "recipe": {**RECIPE, "epochs": args.epochs, "seed": args.seed, "workers": 0, "fraction": args.fraction,
                        "pretrained": False, "resume": False}, "environment": runtime_environment(),
             "checkpoint_precision": "Native trainer stores FP16 EMA; evaluated after FP32 reload. resume.pt preserves optimizer."}

    def save():
        state["updated_utc"] = now()
        write_atomic(run / "experiment.json", state)

    save()
    try:
        wrapper = RTDETR(str(MODELS[args.variant]))
        FactorialHealth().attach(wrapper)

        def started(trainer):
            if (trainer.save_dir.resolve() != run or trainer.batch_size != 8 or trainer.amp
                    or trainer.args.seed != args.seed or trainer.args.workers != 0 or trainer.args.pretrained):
                raise RuntimeError("Trainer changed the owned run, FP32/batch8/seed/workers0/scratch recipe")
            state["architecture"] = inspect_model(unwrap_model(trainer.model), args.variant)
            state["initialization"] = json.loads((run / "initialization.json").read_text())
            if not state["initialization"]["shared_state_verified"]:
                raise RuntimeError("Actual training model lacks paired initialization evidence")
            state["effective_args"] = vars(trainer.args).copy()
            save()

        def epoch_finished(trainer):
            rows, health = read_rows(run / "results.csv"), read_health(run / "health.jsonl")
            finite_numbers(health[-1], location="health")
            state.update(completed_epochs=len(rows), latest_internal_map=float(rows[-1][MAP_KEY]),
                         best_internal_map=max(float(row[MAP_KEY]) for row in rows))
            save()

        wrapper.add_callback("on_train_start", started)
        wrapper.add_callback("on_fit_epoch_end", epoch_finished)
        wrapper.train(trainer=FactorialTrainer, **state["recipe"], data=str(args.data), device="0",
                      project=str(args.project), name=args.name, exist_ok=True)
        rows, health = read_rows(run / "results.csv"), read_health(run / "health.jsonl")
        if (len(rows) != args.epochs or len(health) != args.epochs
                or [int(float(r["epoch"])) for r in rows] != list(range(1, args.epochs + 1))
                or [r["epoch"] for r in health] != list(range(1, args.epochs + 1))):
            raise RuntimeError("Requested epoch budget lacks complete CSV/health evidence")
        for record in health:
            finite_numbers(record, location="health")
            if any(v != 1 for k, v in record.items() if k.endswith("_finite")):
                raise FloatingPointError("Nonfinite bbox updates")
        if any(not math.isfinite(float(v)) for row in rows for v in row.values()):
            raise FloatingPointError("Nonfinite training curve")
        for key in (["p2_first_weight_grad_abs"] if args.variant in ("c2", "c2-la") else []) + (
                ["p5_qkv_grad_abs", "p5_rpb_grad_abs"] if args.variant in ("la", "c2-la") else []):
            if not any((h.get(key) or 0) > 0 for h in health):
                raise RuntimeError("No measured gradient: " + key)
        if source_hashes() != state["source_sha256"] or sha256_file(args.data) != state["data_yaml_sha256"]:
            raise RuntimeError("Frozen source/data YAML changed during training")
        state["checkpoints"] = {}
        for name in ("best", "last", "resume"):
            path = run / "weights" / (name + ".pt")
            state["checkpoints"][name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        restart = torch.load(run / "weights/resume.pt", map_location="cpu", weights_only=False)
        if restart.get("epoch") != args.epochs - 1 or restart.get("optimizer") is None or restart.get("ema") is None:
            raise RuntimeError("Preserved restart asset is incomplete")
        state.update(status="completed", completed_utc=now())
        save()
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=MODELS, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fraction", type=float, default=1.)
    return p.parse_args(argv)


if __name__ == "__main__":
    import fcntl
    with open("/tmp/zhj-uav-5090-gpu0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(parse_args())
