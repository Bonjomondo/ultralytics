"""Train the C0/C2 pair on the 5090 without changing the historical recipe.

Uses the established paired initialization and health collector. A completed
run is a requested-budget result, not proof of convergence or an AP improvement.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs/decoder_budget_5090/ultralytics-config"))

import torch
import yaml

from experiments.r18_p2_sfi.continue_train import ContinuationTrainer
from experiments.r18_p2_sfi.train import MODELS, RECIPE, SFIHealth, build_model
from ultralytics import RTDETR

MAP_KEY = "metrics/mAP50-95(B)"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path, record):
    """Publish a complete state file even if the process exits during a write."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def git(*arguments):
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True).strip()


def sources():
    paths = set((ROOT / "ultralytics").rglob("*.py"))
    paths.update(MODELS[key] for key in ("c0", "c2"))
    paths.update(ROOT / name for name in (
        "train_rtdetr_r18_na.py", "train_paper_repro_visdrone.py",
        "experiments/r18_p2_sfi/train.py", "experiments/r18_p2_sfi/continue_train.py",
        "experiments/r18_p2_sfi/evaluate.py", "experiments/decoder_budget_5090/train_5090.py"))
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sorted(paths)}


def runtime_environment():
    packages = {}
    for name in ("numpy", "scipy", "torchvision", "opencv-python", "polars", "pycocotools", "ultralytics-thop"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    record = {"python": platform.python_version(), "python_executable": sys.executable,
              "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available(),
              "amp": False, "dtype": "float32", "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_tf32": torch.backends.cudnn.allow_tf32, "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "packages": packages}
    if record["cuda_available"]:
        device = torch.cuda.get_device_properties(0)
        record.update(gpu=device.name, capability=list(torch.cuda.get_device_capability(0)),
                      total_memory_bytes=device.total_memory)
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=10)
        record["nvidia_smi"] = {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        record["nvidia_smi"] = {"error": str(error)}
    return record


def read_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def read_health(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def finite_numbers(record, *, location):
    if isinstance(record, dict):
        for key, value in record.items():
            finite_numbers(value, location=f"{location}.{key}")
    elif isinstance(record, list):
        for index, value in enumerate(record):
            finite_numbers(value, location=f"{location}[{index}]")
    elif isinstance(record, (int, float)) and not isinstance(record, bool) and not math.isfinite(record):
        raise FloatingPointError(f"Nonfinite health value: {location}")
    elif record is None and any(token in location for token in ("delta_", "hidden_", "weight_grad_")):
        raise FloatingPointError(f"Missing/nonfinite sampled health value: {location}")


def initialization_record(run, expected_shared):
    record = json.loads((run / "initialization.json").read_text())
    if (record.get("shared_state_verified") is not True or record.get("amp") is not False
            or record.get("seed") != 0 or record.get("shared_state_sha256") != expected_shared):
        raise RuntimeError("Actual trainer initialization differs from the paired seed-0 reference")
    return record


def validate_inputs(args):
    args.data, args.project = args.data.resolve(), args.project.resolve()
    if args.variant not in ("c0", "c2"):
        raise ValueError("Only the preselected C0/C2 pair is supported")
    if "5090" not in args.project.name.lower():
        raise ValueError("--project directory name must contain 5090")
    if not args.name.startswith("5090") or Path(args.name).name != args.name or args.name in (".", ".."):
        raise ValueError("--name must start with 5090 and be a single directory name")
    if args.epochs < 1 or not 0 < args.fraction <= 1 or args.device != "0":
        raise ValueError("Require epochs>=1, fraction in (0,1] and single CUDA device 0")
    if not args.data.is_file() or args.data == (ROOT / "configs/visdrone-local.yaml").resolve():
        raise ValueError("Supply an explicit 5090 data YAML; the historical 4090 data config is not used")
    data = yaml.safe_load(args.data.read_text())
    names = data.get("names", [])
    names = list(names.values()) if isinstance(names, dict) else names
    if (not isinstance(names, list) or len(names) != 10 or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != 10
            or data.get("nc", 10) != 10 or data.get("channels", 3) != 3):
        raise ValueError("This pair requires 10 semantic VisDrone classes and three RGB channels")
    return data


def train(args):
    validate_inputs(args)
    run = args.project / args.name
    # A losing or repeated launch must never write a failure marker into an old run.
    run.mkdir(parents=True, exist_ok=False)
    state_path = run / "experiment_5090.json"
    recipe = {**RECIPE, "epochs": args.epochs}
    state = {"schema": "decoder-budget-5090.training.v1", "status": "running", "phase": "preflight",
             "experiment_name": args.name, "variant": args.variant, "run_directory": str(run),
             "requested_epochs": args.epochs, "fraction": args.fraction, "seed": 0, "current_epoch": 0,
             "experiment_role": "paired_150e_screen" if args.epochs == 150 and args.fraction == 1 else "engineering_smoke_not_architecture_result",
             "started_utc": now(), "updated_utc": now(), "pid": os.getpid(),
             "data_yaml": str(args.data), "recipe": recipe,
             "checkpoint_storage_semantics": "Native trainer stores FP16 EMA and FP16 optimizer tensors; best/last are stripped at completion. resume.pt preserves the unstripped checkpoint. Reloaded evaluation models are converted to FP32; this does not recover discarded storage precision."}

    def save():
        state["updated_utc"] = now()
        write_atomic(state_path, state)

    save()
    try:
        state.update(data_yaml_sha256=sha256_file(args.data), source_commit=git("rev-parse", "HEAD"),
                     source_branch=git("branch", "--show-current"), source_dirty_paths=git("status", "--porcelain").splitlines(),
                     source_sha256=sources())
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        state["environment"] = runtime_environment()
        save()
        if not state["environment"]["cuda_available"] or "5090" not in state["environment"].get("gpu", ""):
            raise RuntimeError("Actual CUDA GPU must be a 5090; this process cannot start the requested experiment")
        expected_shared = build_model(str(MODELS["c0"]), nc=10, ch=3, seed=0).ablation_initialization["shared_state_sha256"]
        state["expected_shared_state_sha256"] = expected_shared
        wrapper = RTDETR(str(MODELS[args.variant]))
        SFIHealth().attach(wrapper)

        def started(trainer):
            if trainer.save_dir.resolve() != run or trainer.batch_size != 8 or trainer.amp:
                raise RuntimeError("Trainer changed the owned directory, batch=8 or FP32 protocol")
            if trainer.args.seed != 0 or trainer.args.workers != 0 or trainer.args.pretrained:
                raise RuntimeError("Trainer changed seed0/workers0/from-scratch initialization")
            state.update(phase="training", initialization=initialization_record(run, expected_shared),
                         environment=runtime_environment(), effective_train_args={key: getattr(trainer.args, key)
                         for key in ("epochs", "batch", "imgsz", "optimizer", "lr0", "lrf", "weight_decay",
                                     "warmup_epochs", "warmup_bias_lr", "nbs", "amp", "deterministic", "seed",
                                     "workers", "fraction", "pretrained", "mosaic", "mixup", "close_mosaic")})
            save()

        def epoch_finished(trainer):
            rows = read_rows(run / "results.csv")
            health = read_health(run / "health.jsonl")
            if not rows or not health:
                raise RuntimeError("Training epoch did not produce CSV and health evidence")
            finite_numbers(health[-1], location="health")
            values = [float(row[MAP_KEY]) for row in rows]
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError("Nonfinite internal mAP history")
            state.update(current_epoch=len(rows), best_internal_map50_95=max(values),
                         latest_internal_map50_95=values[-1], latest_epoch_health=health[-1],
                         train_seconds_total=sum(item["train_seconds"] for item in health))
            save()

        wrapper.add_callback("on_train_start", started)
        wrapper.add_callback("on_fit_epoch_end", epoch_finished)
        wrapper.train(trainer=ContinuationTrainer, **recipe, data=str(args.data), device=args.device,
                      workers=0, fraction=args.fraction, project=str(args.project), name=args.name,
                      pretrained=False, resume=False, exist_ok=True)
        state["phase"] = "completion_checks"
        save()
        rows, health = read_rows(run / "results.csv"), read_health(run / "health.jsonl")
        if len(rows) != args.epochs or [int(float(row["epoch"])) for row in rows] != list(range(1, args.epochs + 1)):
            raise RuntimeError("Trainer did not finish the complete requested epoch budget")
        if len(health) != args.epochs or [row["epoch"] for row in health] != list(range(1, args.epochs + 1)):
            raise RuntimeError("Health evidence does not cover every requested epoch exactly once")
        for row in rows:
            if not all(math.isfinite(float(value)) for value in row.values()):
                raise FloatingPointError("Nonfinite training CSV value")
        for record in health:
            finite_numbers(record, location="health")
            if any(value != 1.0 for key, value in record.items() if key.endswith("delta_finite")):
                raise FloatingPointError("Sampled regression deltas were not finite")
        initialization_record(run, expected_shared)
        if sources() != state["source_sha256"] or sha256_file(args.data) != state["data_yaml_sha256"]:
            raise RuntimeError("Frozen training source or data YAML changed during this run")
        checkpoints = {}
        for name in ("best", "last", "resume"):
            path = run / "weights" / (name + ".pt")
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError(f"Missing completed checkpoint: {path}")
            checkpoints[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        # This file was just generated by this trusted local run. Verify the
        # preserved restart asset before advertising completion to the queue.
        resume = torch.load(run / "weights/resume.pt", map_location="cpu", weights_only=False)
        if resume.get("epoch") != args.epochs - 1 or resume.get("optimizer") is None or resume.get("ema") is None:
            raise RuntimeError("Unstripped resume.pt lacks the expected epoch/optimizer/EMA")
        state.update(status="completed", phase="done", completed_utc=now(), current_epoch=args.epochs,
                     checkpoint_sha256={key: value["sha256"] for key, value in checkpoints.items()},
                     checkpoints=checkpoints, health_epochs=len(health), completed_epochs=len(rows),
                     resume_epoch_zero_based=resume["epoch"])
        save()
        print(json.dumps({"status": "completed", "name": args.name, "epochs": args.epochs, "state": str(state_path)}), flush=True)
        return state
    except BaseException as error:
        state.update(status="failed", phase="failed", failed_utc=now(), error=f"{type(error).__name__}: {error}")
        save()
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("c0", "c2"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--device", choices=("0",), default="0")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
