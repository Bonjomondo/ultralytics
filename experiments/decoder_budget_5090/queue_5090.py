"""Serial 5090 C0/C2 baseline queue; real-data gates precede full training.

Run in tmux with stdout redirected to a log. Existing phase directories and
logs are never reused; --continue-after-gates only continues an intact gate run.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.util import find_spec
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXPECTED_COMMIT = "fc843c1d62162742deac3f72b52c67291b606b8c"
BUDGETS = [300, 250, 200, 150]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sources(sdk, data, annotation):
    files = [*ROOT.joinpath("ultralytics").rglob("*.py"),
             *ROOT.joinpath("ultralytics/cfg").rglob("*.yaml"),
             *HERE.glob("*.py"), *ROOT.joinpath("experiments/r18_p2_sfi").glob("*.py"),
             *sdk.joinpath("src/detrscope").rglob("*.py"),
             ROOT / "train_rtdetr_r18_na.py", ROOT / "train_paper_repro_visdrone.py", data, annotation]
    return {str(p.resolve()): digest(p) for p in sorted(set(files))}


def validate_training(run, epochs, variant):
    record = read(run / "experiment_5090.json")
    if record["status"] != "completed" or record["current_epoch"] != epochs:
        raise RuntimeError(f"Training did not complete {epochs} epochs: {run}")
    with (run / "results.csv").open() as stream:
        curve = list(csv.DictReader(stream))
    if len(curve) != epochs or int(curve[-1]["epoch"]) != epochs:
        raise RuntimeError(f"Incomplete training curve: {run}")
    for row in curve:
        for key, value in row.items():
            if ("loss" in key or "metrics/" in key) and not math.isfinite(float(value)):
                raise RuntimeError(f"Nonfinite {key}: {run}")
    health = [json.loads(line) for line in (run / "health.jsonl").read_text().splitlines()]
    if len(health) != epochs or [r["epoch"] for r in health] != list(range(1, epochs + 1)):
        raise RuntimeError(f"Incomplete health observations: {run}")
    for row in health:
        for key, value in row.items():
            if key.endswith("_finite") and value != 1.0:
                raise RuntimeError(f"Nonfinite tensor health: {run}: {key}")
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise RuntimeError(f"Nonfinite health scalar: {run}: {key}")
    if variant == "c2" and not any((r.get("p2_first_weight_grad_abs") or 0) > 0 for r in health):
        raise RuntimeError(f"C2 P2 branch has no observed nonzero gradient: {run}")
    initialization = read(run / "initialization.json")
    if not initialization["shared_state_verified"]:
        raise RuntimeError(f"Unverified paired initialization: {run}")
    checkpoints = {name: digest(run / "weights" / name) for name in ("best.pt", "last.pt", "resume.pt")}
    return {"run": str(run), "epochs": epochs, "shared_state_sha256": initialization["shared_state_sha256"],
            "checkpoint_sha256": checkpoints, "status_sha256": digest(run / "experiment_5090.json")}


def validate_sweep(run, images):
    result = read(run / "summary.json")
    rows = result["conditions"]
    if result["status"] != "completed" or [r["queries"] for r in rows] != BUDGETS:
        raise RuntimeError(f"Incomplete four-budget sweep: {run}")
    if any(r["images"] != images for r in rows) or len({r["input_gt_identity_sha256"] for r in rows}) != 1:
        raise RuntimeError(f"Sweep input/GT/image-count identity differs: {run}")
    return {"run": str(run), "images": images, "summary_sha256": digest(run / "summary.json")}


class Queue:
    def __init__(self, args):
        self.args = args
        self.root, self.sdk = args.root.resolve(), args.sdk.resolve()
        self.data_root = args.data_root.resolve()
        self.data = self.root / "data_5090.yaml"
        self.annotation = self.data_root / "annotations/VisDrone2019-DET_val_coco.json"
        self.pin = self.root / "source_pin_5090.json"
        self.status = self.root / "status_5090.json"
        self.frozen = self.root / "source_sha256_5090.json"
        self.gates = self.root / "gates_passed_5090.json"
        self.child = None
        self.owned = False
        self.state = {}
        self.environment = {**os.environ, "PYTHONPATH": f"{ROOT}:{self.sdk / 'src'}", "MPLBACKEND": "Agg",
                            "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4",
                            "YOLO_CONFIG_DIR": str(self.root / "config_5090")}
        if "5090" not in self.root.name:
            raise ValueError("Experiment root name must contain 5090")

    def check_sources(self):
        if sources(self.sdk, self.data, self.annotation) != read(self.frozen):
            raise RuntimeError("Frozen source/data/annotation identity changed; refusing the next phase")

    def update(self, **values):
        self.state.update(values, updated_utc=now())
        save(self.status, self.state)

    def prepare(self):
        if self.args.continue_after_gates:
            self.state = read(self.status)
            if self.state["status"] != "gates_passed":
                raise RuntimeError("Only a successfully finished --gate-only run can continue")
            self.check_sources()
            receipt = read(self.gates)
            if receipt["source_pin_sha256"] != digest(self.pin):
                raise RuntimeError("Gate source pin changed")
            training, sweeps = self.verify_gates()
            if receipt["training"] != training or receipt["sweeps"] != sweeps:
                raise RuntimeError("Gate evidence changed since its successful validation")
            self.owned = True
            self.update(status="running", pid=os.getpid(), continued_utc=now())
            return
        self.root.mkdir(parents=True, exist_ok=False)
        self.owned = True
        import yaml
        original = yaml.safe_load((self.data_root / "data.yaml").read_text())
        data = {"path": str(self.data_root), "train": "images/train", "val": "images/val",
                "nc": 10, "names": original["names"]}
        self.data.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        for path in (self.annotation, self.data_root / "images/train", self.data_root / "images/val"):
            if not path.exists():
                raise FileNotFoundError(path)
        if len(read(self.annotation)["images"]) != 548:
            raise RuntimeError("Expected the frozen 548-image VisDrone validation set")
        save(self.frozen, sources(self.sdk, self.data, self.annotation))
        self.state = {"status": "running", "started_utc": now(), "pid": os.getpid(), "completed_phases": [],
                      "scope": "fresh 5090 seed0 C0/C2 paired baselines; historical 4090 AP is reference only"}
        self.update()
        self.phase("5090-source-pin", [sys.executable, str(HERE / "run.py"), "--sdk", str(self.sdk), "pin",
                                      "--expected-commit", EXPECTED_COMMIT, "--output", str(self.pin)], self.pin)

    def phase(self, name, command, output, validator=None):
        self.check_sources()
        log = self.root / "logs_5090" / (name + ".log")
        if output.exists() or log.exists():
            raise FileExistsError(f"Refusing to reuse phase output/log: {name}")
        log.parent.mkdir(exist_ok=True)
        self.update(status="running", phase=name, name=name, command=command, log=str(log), child_pid=None)
        with log.open("x") as stream:
            self.child = subprocess.Popen(command, cwd=ROOT, env=self.environment, stdout=stream, stderr=subprocess.STDOUT)
            self.update(child_pid=self.child.pid)
            returncode = self.child.wait()
        self.child = None
        if returncode:
            raise RuntimeError(f"{name} exited {returncode}; inspect {log}")
        if validator:
            validator(output)
        self.state["completed_phases"].append(name)
        self.update(child_pid=None)

    def train(self, variant, gate):
        epochs = 1 if gate else 150
        suffix = "1e-gate" if gate else "150e"
        name = f"5090-rtdetr-r18-{variant}-{suffix}-s0"
        output = self.root / name
        command = [sys.executable, "-u", str(HERE / "train_5090.py"), "--variant", variant, "--data", str(self.data),
                   "--project", str(self.root), "--name", name, "--epochs", str(epochs),
                   "--fraction", "0.01" if gate else "1.0", "--device", "0"]
        self.phase(name, command, output, lambda p: validate_training(p, epochs, variant))
        return output

    def sweep(self, training, suffix, full=False):
        name = training.name + f"-query-{suffix}"
        output = self.root / name
        command = [sys.executable, "-u", str(HERE / "run.py"), "--sdk", str(self.sdk), "sweep",
                   "--pin", str(self.pin), "--checkpoint", str(training / "weights/best.pt"),
                   "--annotations", str(self.annotation), "--images", str(self.data_root / "images/val"),
                   "--device", "cuda:0", "--expected-gpu", "5090", "--output", str(output),
                   "--limit", "0" if full else "8", "--batch", "2"]
        if full:
            command.append("--benchmark")
        self.phase(name, command, output, lambda p: validate_sweep(p, 548 if full else 8))
        return output

    def verify_gates(self):
        training, sweeps = {}, {}
        for variant in ("c0", "c2"):
            run = self.root / f"5090-rtdetr-r18-{variant}-1e-gate-s0"
            training[variant] = validate_training(run, 1, variant)
            sweeps[variant] = validate_sweep(self.root / (run.name + "-query-gate"), 8)
        if training["c0"]["shared_state_sha256"] != training["c2"]["shared_state_sha256"]:
            raise RuntimeError("C0/C2 shared initialization differs")
        return training, sweeps

    def run(self):
        try:
            self.prepare()
            if not self.args.continue_after_gates:
                for variant in ("c0", "c2"):
                    training = self.train(variant, gate=True)
                    self.sweep(training, "gate")
                training, sweeps = self.verify_gates()
                save(self.gates, {"status": "passed", "finished_utc": now(), "training": training, "sweeps": sweeps,
                                  "source_pin_sha256": digest(self.pin), "scope": "engineering gate; one epoch is not an AP comparison"})
                self.update(status="gates_passed", phase="5090-gates-passed", finished_utc=now())
                if self.args.gate_only:
                    return
            for variant in ("c0", "c2"):
                training = self.train(variant, gate=False)
                self.sweep(training, "gate")
                self.sweep(training, "full", full=True)
            self.update(status="completed", phase="5090-completed", child_pid=None, finished_utc=now())
        except BaseException as error:
            if self.child is not None and self.child.poll() is None:
                self.child.terminate()
                try:
                    self.child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait()
            if self.owned:
                self.update(status="failed", error=f"{type(error).__name__}: {error}", child_pid=None, finished_utc=now())
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, default=ROOT.parent / "detrscope")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "SO_DETR/datasets/VisDrone2019-DET-YOLO")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--gate-only", action="store_true")
    choice.add_argument("--continue-after-gates", action="store_true")
    args = parser.parse_args()
    required = ("torch", "torchvision", "scipy", "numpy", "cv2", "yaml", "polars", "pycocotools", "matplotlib")
    missing = [name for name in required if find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing experiment dependencies before launch: {', '.join(missing)}")
    queue = Queue(args)
    with Path("/tmp/zhj-uav-5090-gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0):
            raise RuntimeError("This queue requires the actual 5090 CUDA device")
        queue.run()


if __name__ == "__main__":
    main()
