"""Run the four pre-registered 150-epoch arms serially, stopping on any failed phase."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
RUNS = ROOT / "runs/detect/r18_p2_sfi_20261023"
ORDER = ["c0", "c3", "c2", "c1"]
STATUS = OUTPUT / "queue_status.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(state):
    state["updated_utc"] = now()
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STATUS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-completed-phases", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.update(PYTHONPATH=str(ROOT), MPLBACKEND="Agg", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4")
    gpu_lock = Path("/tmp/r18-p2-sfi-gpu0.lock").open("a")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = [*sorted((ROOT / "ultralytics").rglob("*.py")),
               *sorted((ROOT / "ultralytics/cfg").rglob("*.yaml")),
               *sorted(OUTPUT.glob("*.py")), ROOT / "train_rtdetr_r18_na.py",
               ROOT / "train_paper_repro_visdrone.py", ROOT / "configs/visdrone-local.yaml"]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    manifest = OUTPUT / "source_sha256.json"
    if STATUS.exists():
        if not args.resume_completed_phases:
            raise FileExistsError("Queue state exists; refusing an accidental second launch")
        state = json.loads(STATUS.read_text())
        if state["order"] != ORDER or json.loads(manifest.read_text()) != hashes:
            raise RuntimeError("Queue identity or frozen sources changed")
        state.update(status="running", pid=os.getpid())
    else:
        if args.resume_completed_phases:
            raise FileNotFoundError(STATUS)
        RUNS.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(hashes, indent=2) + "\n")
        with tarfile.open(RUNS / "source_snapshot.tar.gz", "w:gz") as archive:
            for path in sources:
                archive.add(path, arcname=str(path.relative_to(ROOT)))
        state = {"status": "running", "started_utc": now(), "pid": os.getpid(), "order": ORDER,
                 "completed_training": [], "completed_evaluation": [], "runs": {}}
    save(state)
    try:
        for variant in ORDER:
            name = f"rtdetr-r18-{variant}-150e-s0"
            run = RUNS / name
            commands = [("training", [sys.executable, "-u", str(OUTPUT / "train.py"),
                         "--variant", variant, "--project", str(RUNS), "--name", name]),
                        ("evaluation", [sys.executable, "-u", str(OUTPUT / "evaluate.py"),
                         "--variant", variant, "--run", str(run)])]
            for phase, command in commands:
                for path, expected in hashes.items():
                    if digest(ROOT / path) != expected:
                        raise RuntimeError(f"Frozen source changed: {path}")
                if variant in state[f"completed_{phase}"]:
                    continue
                if phase == "training" and run.exists():
                    raise FileExistsError(f"Partial run exists; explicitly resume its last.pt first: {run}")
                log = ROOT / "logs" / f"p2-sfi-{variant}-{phase}-20261023.log"
                if log.exists():
                    log = log.with_name(f"{log.stem}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.log")
                state.update(variant=variant, phase=phase, run=str(run), log=str(log), command=command)
                state["runs"][variant] = str(run)
                save(state)
                with log.open("x") as file:
                    child = subprocess.Popen(command, stdout=file, stderr=subprocess.STDOUT)
                    state["child_pid"] = child.pid
                    save(state)
                    rc = child.wait()
                if rc:
                    raise RuntimeError(f"{variant} {phase} exited {rc}; inspect {log}")
                if phase == "training":
                    with (run / "results.csv").open() as file:
                        rows = list(csv.DictReader(file))
                    if len(rows) != 150 or int(rows[-1]["epoch"]) != 150:
                        raise RuntimeError("Trainer did not finish the full 150 epochs")
                else:
                    record = json.loads((OUTPUT / f"{variant}_formal_metrics.json").read_text())
                    if record["checkpoint_sha256"] != digest(run / "weights/best.pt"):
                        raise RuntimeError("Evaluation checkpoint identity mismatch")
                state[f"completed_{phase}"].append(variant)
                save(state)
        state.update(status="completed", phase="done", child_pid=None, finished_utc=now())
        save(state)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", failed_utc=now())
        save(state)
        raise


if __name__ == "__main__":
    main()
