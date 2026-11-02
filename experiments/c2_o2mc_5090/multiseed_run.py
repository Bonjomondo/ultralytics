"""Audited smoke -> C2+O2M-C 150e -> native export -> full COCO comparison."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import traceback

import torch
import yaml

from experiments.c2_o2mc_5090.model import O2MCTrainer, strip_auxiliary
from experiments.decoder_budget_5090.train_5090 import (
    finite_numbers, git, now, read_health, read_rows, runtime_environment, sha256_file, sources, write_atomic,
)
from experiments.p2_p5_la_5090.evaluate import coco_metrics, category_ids, native_predictor
from experiments.p2_p5_la_5090.model import inspect_model
from experiments.r18_p2_sfi.continue_train import plateau_status
from experiments.r18_p2_sfi.train import MODELS, RECIPE, tensor_digest
from train_rtdetr_r18_na import TrainingHealth
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import unwrap_model

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT.parent / "SO_DETR/datasets/VisDrone2019-DET-YOLO"
ANN = DATA / "annotations/VisDrone2019-DET_val_coco.json"
BASELINE = ROOT / "runs/decoder_budget_5090/5090-c0c2-baselines-20261024-s0-r1/5090-rtdetr-r18-c2-150e-s0/weights/best.pt"
MAP = "metrics/mAP50-95(B)"


def source_hashes():
    result = sources()
    for directory in (HERE, ROOT / "experiments/p2_p5_la_5090"):
        result.update({p.relative_to(ROOT).as_posix(): sha256_file(p) for p in directory.glob("*.py")})
    result.update({p.relative_to(ROOT).as_posix(): sha256_file(p) for p in HERE.glob("*.yaml")})
    return result


def configure():
    import polars  # Required for native checkpoint CSV serialization; fail before training.
    import ultralytics
    import detrscope
    expected = Path("/home/ubuntu/miniconda3/envs/UAV_CV")
    if Path(sys.executable) != expected / "bin/python" or Path(sys.prefix) != expected or Path(sys.base_prefix) != expected:
        raise RuntimeError("Use conda UAV_CV directly")
    if Path(ultralytics.__file__).resolve() != ROOT / "ultralytics/__init__.py":
        raise RuntimeError("Wrong ultralytics implementation")
    if Path(detrscope.__file__).resolve() != ROOT.parent / "detrscope/src/detrscope/__init__.py":
        raise RuntimeError("Wrong DETRScope implementation")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Require CUDA:0 RTX 5090")
    return {**runtime_environment(), "sys_prefix": sys.prefix, "sys_base_prefix": sys.base_prefix,
            "ultralytics_file": ultralytics.__file__, "detrscope_file": detrscope.__file__,
            "PYTHONPATH": os.environ.get("PYTHONPATH"), "cwd": str(Path.cwd()), "pid": os.getpid()}


def verify_plan(root):
    plan = json.loads((root / "plan.json").read_text())
    if plan["source_sha256"] != source_hashes() or plan["annotations_sha256"] != sha256_file(ANN):
        raise RuntimeError("Frozen source/config/annotations changed; preserve this run and audit")
    if plan["data_yaml_sha256"] != sha256_file(root / "data.yaml"):
        raise RuntimeError("Frozen dataset configuration changed")
    return plan


def train(args):
    root, name = args.root.resolve(), args.name
    plan = verify_plan(root)
    run = root / "training" / name
    run.mkdir(parents=True, exist_ok=False)
    state = {"schema": "5090.c2-o2mc.training.v1", "status": "running", "started_utc": now(),
        "pid": os.getpid(), "variant": "c2-o2mc", "seed": plan["seed"], "completed_epochs": 0,
        "role": "engineering_smoke" if args.smoke else "single_seed_150e_screen",
        "source_commit": git("rev-parse", "HEAD"), "source_branch": git("branch", "--show-current"),
        "source_sha256": plan["source_sha256"], "environment": configure(),
        "recipe": {**RECIPE, "epochs": 1 if args.smoke else 150, "seed": plan["seed"], "workers": 0,
                   "fraction": .01 if args.smoke else 1., "pretrained": False, "resume": False,
                   "save_period": -1},
        "auxiliary": yaml.safe_load((HERE / "recipe.yaml").read_text())["auxiliary"],
        "checkpoint_storage": "FP32 training; native trainer stores FP16 EMA/optimizer. resume.pt preserves unstripped final state."}

    def save():
        state["updated_utc"] = now()
        write_atomic(run / "experiment.json", state)

    save()
    try:
        wrapper = RTDETR(str(MODELS["c2"]))
        TrainingHealth().attach(wrapper)
        sampled = set()
        batch_index = 0

        def started(trainer):
            model = unwrap_model(trainer.model)
            if (trainer.batch_size != 8 or trainer.amp or trainer.args.pretrained or trainer.args.seed != plan["seed"]
                    or trainer.save_dir.resolve() != run or not model.ablation_initialization["native_c2_state_verified"]):
                raise RuntimeError("Actual trainer violated the controlled experiment")
            if model.ablation_initialization["c2_full_state_sha256"] != plan["paired_initialization"]["native_full_state_sha256"]:
                raise RuntimeError("Native initialization differs from the historical paired C2")
            if model.ablation_initialization["r0_full_state_sha256"] != plan["paired_initialization"]["r0_full_state_sha256"]:
                raise RuntimeError("Historical reference initialization changed")
            if (model.aux_weight != state["auxiliary"]["weight"]
                    or model.switch_epoch != state["auxiliary"]["switch_epoch_zero_based"]):
                raise RuntimeError("Model and experiment auxiliary configuration disagree")
            state.update(initialization=model.ablation_initialization, architecture=inspect_model(model, "c2"),
                         effective_args=vars(trainer.args).copy(),
                         train_images=len(trainer.train_loader.dataset), val_images=len(trainer.test_loader.dataset),
                         batches_per_epoch=len(trainer.train_loader),
                         warmup_iterations=max(round(trainer.args.warmup_epochs * len(trainer.train_loader)), 100),
                         gradient_accumulation={"micro_batch": 8, "cards": 1, "after_warmup": 8,
                             "effective_batch_after_warmup": 64, "warmup": "linear interpolation 1->8 per training iteration",
                             "batchnorm": "micro batch statistics; not equivalent to batch64"})
            state["architecture"].update(native_parameters=sum(p.numel() for p in model.model.parameters()),
                                         auxiliary_parameters=sum(p.numel() for p in model.aux_head.parameters()))
            save()

        def epoch_start(trainer):
            nonlocal batch_index
            batch_index = 0
            model = unwrap_model(trainer.model)
            model.train_epoch = trainer.epoch
            model.capture_diagnostics = False

        def batch_start(trainer):
            unwrap_model(trainer.model).capture_diagnostics = batch_index == 2

        def batch_end(trainer):
            nonlocal batch_index
            model = unwrap_model(trainer.model)
            if trainer.batch_size != 8 or not torch.isfinite(trainer.loss).all():
                raise RuntimeError("Nonfinite loss or automatic batch reduction; preserve failed run")
            if model.capture_diagnostics:
                record = {"epoch": trainer.epoch + 1, "batch_index_zero_based": batch_index,
                          **model.latest_diagnostics}
                finite_numbers(record, location="aux_diagnostics")
                with (run / "aux_diagnostics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
            model.capture_diagnostics = False
            batch_index += 1

        def finished(trainer):
            epoch = trainer.epoch + 1
            if epoch in sampled:
                return
            sampled.add(epoch)
            verify_plan(root)
            rows, health = read_rows(run / "results.csv"), read_health(run / "health.jsonl")
            finite_numbers(health[-1], location="health")
            values = [float(row[MAP]) for row in rows]
            for row in rows:
                finite_numbers({k: float(v) for k, v in row.items()}, location="training_csv")
            state.update(completed_epochs=len(rows), latest_internal_map=values[-1], best_internal_map=max(values),
                         convergence=plateau_status(values), latest_health=health[-1])
            save()

        for event, callback in (("on_train_start", started), ("on_train_epoch_start", epoch_start),
            ("on_train_batch_start", batch_start), ("on_train_batch_end", batch_end), ("on_fit_epoch_end", finished)):
            wrapper.add_callback(event, callback)
        wrapper.train(trainer=O2MCTrainer, **state["recipe"], data=str(root / "data.yaml"), device="0",
                      project=str(run.parent), name=name, exist_ok=True)
        rows = read_rows(run / "results.csv")
        if len(rows) != state["recipe"]["epochs"] or len(read_health(run / "health.jsonl")) != len(rows):
            raise RuntimeError("Incomplete training curve or health evidence")
        if len(read_health(run / "aux_diagnostics.jsonl")) != len(rows):
            raise RuntimeError("Missing per-epoch auxiliary/gradient diagnostics")
        state["checkpoints"] = {key: {"path": str(run / "weights" / f"{key}.pt"),
            "sha256": sha256_file(run / "weights" / f"{key}.pt"),
            "bytes": (run / "weights" / f"{key}.pt").stat().st_size} for key in ("best", "last", "resume")}
        state.update(status="completed", finished_utc=now())
        save()
    except BaseException:
        state.update(status="failed", finished_utc=now(), error=traceback.format_exc())
        save()
        raise


@torch.inference_mode()
def evaluate(args):
    import cv2
    import numpy as np
    from pycocotools.coco import COCO
    configure()
    plan = verify_plan(args.root)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = (saved.get("ema") or saved["model"]).float().eval()
    architecture = inspect_model(model, "c2")
    initial = tensor_digest(model.state_dict())
    native = strip_auxiliary(model) if hasattr(model, "aux_head") else model
    native = native.cuda().eval()
    native_initial = tensor_digest(native.state_dict())
    gt = COCO(str(ANN))
    names = [native.names[i] for i in range(10)]
    categories = category_ids(gt, names)
    images = sorted(gt.dataset["images"], key=lambda item: item["id"])
    if len(images) != 548 or len({im["id"] for im in images}) != 548:
        raise ValueError("Require the complete established validation split")
    if args.smoke:
        images = images[:8]
    predictor = native_predictor(native.names)
    protocol = {"dtype": "float32", "fused": False, "batch": 2, "imgsz": 640, "conf": .001,
                "prediction_max_det": 300, "formal_max_dets": [1, 10, 100], "nms": False,
                "annotations_sha256": plan["annotations_sha256"], "image_ids": [im["id"] for im in images],
                "category_ids": categories, "preprocess": "native scale_fill, RGB BCHW /255",
                "area": "COCO APs uses original annotation area; training diagnostic uses resized area"}
    predictions, equality = [], False
    for start in range(0, len(images), 2):
        selected = images[start:start + 2]
        paths = [DATA / "images/val" / Path(im["file_name"]).name for im in selected]
        originals = [cv2.imread(str(p)) for p in paths]
        if any(im is None for im in originals):
            raise FileNotFoundError(str(paths))
        for meta, image in zip(selected, originals):
            if image.shape[:2] != (meta["height"], meta["width"]):
                raise ValueError("Image dimensions disagree with GT")
        array = np.stack(predictor.pre_transform(originals))[..., ::-1].copy()
        inputs = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous().cuda().float() / 255
        output = native(inputs)[0]
        if not torch.isfinite(output).all() or output.shape != (len(selected), 300, 14):
            raise RuntimeError("Invalid native C2 predictions")
        if start == 0 and hasattr(model, "aux_head"):
            model.cuda()
            def forbidden(*unused):
                raise RuntimeError("Auxiliary branch executed during eval")
            hook = model.aux_head.register_forward_pre_hook(forbidden)
            try:
                torch.testing.assert_close(model(inputs)[0], output, atol=0, rtol=0)
            finally:
                hook.remove()
            equality = True
            model.cpu()
        predictor.batch = ([str(p) for p in paths], originals, [""] * len(paths))
        results = predictor.postprocess((output, None), None, originals)
        for meta, result in zip(selected, results):
            for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                x1, y1, x2, y2 = box
                predictions.append({"image_id": meta["id"], "category_id": categories[int(cls)],
                                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    metrics, per_class = coco_metrics(gt, predictions, protocol["image_ids"])
    write_atomic(out / "predictions.json", predictions)
    if tensor_digest(model.state_dict()) != initial or tensor_digest(native.state_dict()) != native_initial:
        raise RuntimeError("Evaluation mutated model state")
    if hasattr(model, "aux_head"):
        torch.save({"model": native.cpu(), "ema": None, "train_args": saved["train_args"],
                    "source_checkpoint_sha256": sha256_file(args.checkpoint)}, out / "native-c2.pt")
    record = {"status": "completed", "scope": "engineering_8_images" if args.smoke else "full_548_images",
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256_file(args.checkpoint)},
        "protocol": protocol, "metrics": metrics, "per_class": per_class, "architecture": architecture,
        "native_export_exact": equality, "model_state_unchanged": True,
        "predictions_sha256": sha256_file(out / "predictions.json"), "finished_utc": now()}
    write_atomic(out / "summary.json", record)
    print(json.dumps(record["metrics"], indent=2), flush=True)


def pipeline(args):
    root = args.root.resolve()
    seed = args.seed
    if seed not in (1, 2):
        raise ValueError("Only authorized seeds 1 and 2")
    baseline = ROOT / f"runs/p2_stability_5090/5090-p2-stability-20261027-r1/training/5090-c2-150e-s{seed}/weights/best.pt"
    from experiments.c2_o2mc_5090.multiseed_support import paired_initialization, check_reference
    reference = check_reference()
    paired = paired_initialization(seed, baseline)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise FileExistsError("This pipeline never overwrites or resumes an existing experiment")
    (root / "logs").mkdir(exist_ok=True)
    shutil.copy2(HERE / "data.yaml", root / "data.yaml")
    plan = {"experiment": "C2 + O2M-C", "created_utc": now(), "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"), "dirty_paths": git("status", "--porcelain").splitlines(),
        "environment": configure(), "source_sha256": source_hashes(), "data_yaml_sha256": sha256_file(root / "data.yaml"),
        "annotations_path": str(ANN), "annotations_sha256": sha256_file(ANN),
        "seed": seed, "paired_initialization": paired, "reference_seed0": reference,
        "archive_output": str(args.output.resolve()),
        "baseline_checkpoint": {"path": str(baseline), "sha256": sha256_file(baseline)},
        "recipe": {**yaml.safe_load((HERE / "recipe.yaml").read_text()), "seed": seed},
        "tmux": os.environ.get("TMUX"), "launch": str(root / "launch.sh")}
    write_atomic(root / "plan.json", plan)
    with tarfile.open(root / "source_snapshot.tar.gz", "x:gz") as archive:
        for relative in plan["source_sha256"]:
            archive.add(ROOT / relative, arcname=relative)
    (root / "source.patch").write_text(git("diff", "HEAD") + "\n")
    state = {"status": "running", "pid": os.getpid(), "started_utc": now(), "stages": []}

    def stage(name, arguments):
        verify_plan(root)
        log_path = root / "logs" / f"{name}.log"
        record = {"name": name, "command": [sys.executable, "-u", *arguments],
                  "log": str(log_path), "started_utc": now(), "status": "running"}
        state["stages"].append(record)
        state["phase"] = name
        with log_path.open("x") as stream:
            process = subprocess.Popen(record["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            record["pid"] = process.pid
            write_atomic(root / "status.json", state)
            code = process.wait()
        record.update(status="completed" if code == 0 else "failed", returncode=code, finished_utc=now())
        write_atomic(root / "status.json", state)
        if code:
            raise RuntimeError(f"{name} failed ({code}): {log_path}")

    prefix = ["-m", "experiments.c2_o2mc_5090.multiseed_run"]
    try:
        stage("dataset-audit", [str(ROOT / "docs/experiment_reports/2026-10-24_5090_baselines/audit_assets_5090.py"),
                                "--output", str(root / "asset_audit.json")])
        audit = json.loads((root / "asset_audit.json").read_text())
        old = json.loads((ROOT / "docs/experiment_reports/2026-10-24_5090_baselines/raw/asset_audit_5090.json").read_text())
        if audit["status"] != "passed" or any(audit[k] != old[k] for k in ("label_content_sha256", "image_inventory_sha256")):
            raise RuntimeError("Data identity differs from the historical C2 baseline")
        stage("cpu-gates", ["-m", "experiments.c2_o2mc_5090.test_o2mc"])
        stage("smoke-train", [*prefix, "train", "--root", str(root), "--name", f"5090-c2-o2mc-smoke-s{seed}", "--smoke"])
        stage("smoke-eval", [*prefix, "evaluate", "--root", str(root), "--smoke",
            "--checkpoint", str(root / f"training/5090-c2-o2mc-smoke-s{seed}/weights/best.pt"), "--output", str(root / "smoke-evaluation")])
        if args.smoke:
            state.update(status="smoke_completed", finished_utc=now())
            write_atomic(root / "status.json", state)
            return
        stage("full-train", [*prefix, "train", "--root", str(root), "--name", f"5090-c2-o2mc-150e-s{seed}"])
        stage("full-eval", [*prefix, "evaluate", "--root", str(root),
            "--checkpoint", str(root / f"training/5090-c2-o2mc-150e-s{seed}/weights/best.pt"), "--output", str(root / "evaluation")])
        stage("baseline-eval", [*prefix, "evaluate", "--root", str(root),
            "--checkpoint", str(baseline), "--output", str(root / "baseline-evaluation")])
        a = json.loads((root / "evaluation/summary.json").read_text())
        b = json.loads((root / "baseline-evaluation/summary.json").read_text())
        if a["protocol"] != b["protocol"]:
            raise RuntimeError("Candidate/baseline evaluation protocol mismatch")
        comparison = {"candidate": a["metrics"], "baseline": b["metrics"],
            "delta_pp": {key: 100 * (a["metrics"][key] - b["metrics"][key]) for key in a["metrics"]},
            "boundary": "Single seed, fixed 150e budget. No multi-seed stability or C0/Aux synergy conclusion."}
        write_atomic(root / "comparison.json", comparison)
        stage("archive-final", ["-m", "experiments.c2_o2mc_5090.multiseed_archive", "--root", str(root),
            "--output", str(args.output.resolve())])
        state.update(status="completed", finished_utc=now(), comparison=comparison)
        write_atomic(root / "status.json", state)
    except BaseException:
        state.update(status="failed", error=traceback.format_exc(), finished_utc=now())
        write_atomic(root / "status.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pipeline", "train", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, choices=(1, 2))
    parser.add_argument("--name")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.action == "train":
        if not args.name or Path(args.name).name != args.name or not args.name.startswith("5090-"):
            parser.error("train requires one --name beginning 5090-")
        train(args)
    elif args.action == "evaluate":
        if args.checkpoint is None or args.output is None:
            parser.error("evaluate requires --checkpoint and --output")
        evaluate(args)
    else:
        if args.seed is None or args.output is None:
            parser.error("pipeline requires --seed and --output")
        with open("/tmp/zhj-uav-5090-gpu0.lock", "a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pipeline(args)


if __name__ == "__main__":
    main()
