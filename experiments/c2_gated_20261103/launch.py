"""Freeze committed sources and start one detached GPU queue under tmux."""
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from common import ARMS, EXP, REPO, REPORT, ROOT, RUNS, now, sha, write


def main():
    assert REPO == ROOT, "Launch from the actual repository, not an existing source snapshot"
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    assert branch == "feat/r18-c2-gated-fusion"
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip(), "Commit reviewed work first"
    proc = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    assert not proc.strip(), "GPU has another compute process; do not launch a conflicting experiment"
    tmux = "/home/zjnu/anaconda3/bin/tmux"
    assert subprocess.run([tmux, "has-session", "-t", "c2-gated-20261103"], capture_output=True).returncode != 0
    assert not (RUNS / "queue_status.json").exists(), "Queue already launched; inspect it instead of duplicating"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    frozen = RUNS / "source"
    assert not frozen.exists()
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO).decode().split("\0")
    manifest = {}
    for rel in filter(None, files):
        src, dst = REPO / rel, frozen / rel
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if src.suffix in [".py", ".yaml", ".yml"]:
                manifest[rel] = sha(dst)
    write(RUNS / "source_manifest.json", manifest)
    write(REPORT / "raw/source_identity.json", {"branch": branch, "commit": commit,
          "frozen_source": str(frozen), "source_files": len(manifest),
          "manifest": str(RUNS / "source_manifest.json"), "manifest_sha256": sha(RUNS / "source_manifest.json")})
    # A nonexistent YOLO_CONFIG_DIR silently falls back to /tmp/Ultralytics.
    # Pre-create the private directory and seed the existing font to avoid a network download.
    config = RUNS / "yolo_config"
    config.mkdir(parents=True, exist_ok=True)
    font = Path.home() / ".config/Ultralytics/Arial.ttf"
    if not font.exists():
        raise RuntimeError("Expected preinstalled Arial.ttf; resolve fonts before starting GPU jobs")
    shutil.copyfile(font, config / "Arial.ttf")
    environment = {
        "PYTHONPATH": str(frozen), "C2_GATED_REPO_ROOT": str(REPO),
        "YOLO_AUTOINSTALL": "false", "YOLO_CONFIG_DIR": str(RUNS / "yolo_config"),
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
        "MPLBACKEND": "Agg", "WANDB_MODE": "disabled", "COMET_MODE": "DISABLED", "PYTHONFAULTHANDLER": "1",
    }
    launcher = RUNS / "launch.sh"
    command = ["env", *[f"{k}={v}" for k, v in environment.items()], sys.executable,
               "-u", str(frozen / "experiments/c2_gated_20261103/controller.py")]
    launcher.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(frozen)) +
                        "\nexec " + shlex.join(command) + " > " + shlex.quote(str(RUNS / "controller.log")) + " 2>&1\n")
    subprocess.run([tmux, "new-session", "-d", "-s", "c2-gated-20261103", "bash", str(launcher)], check=True)
    write(REPORT / "raw/launch.json", {"launched_utc": now(), "branch": branch, "source_commit": commit,
          "tmux": "c2-gated-20261103", "arms": list(ARMS), "epochs_each": 150,
          "source_snapshot": str(frozen), "queue_status": str(RUNS / "queue_status.json"),
          "controller_log": str(RUNS / "controller.log"), "environment": environment})
    print("LAUNCHED", str(RUNS), commit, flush=True)


if __name__ == "__main__":
    main()
