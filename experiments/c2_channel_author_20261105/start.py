"""Freeze the committed author-engine experiment and launch one tmux queue."""
import argparse
import json
import shlex
import shutil
import subprocess
import sys

from common import EXP, REPO, REPORT, RUNS, now, sha, write


def main():
    p = argparse.ArgumentParser();p.add_argument("--resume-controller", action="store_true");args = p.parse_args()
    tmux = "/home/zjnu/anaconda3/bin/tmux";session = "c2-channel-author-20261105"
    assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip(), "GPU busy"
    assert subprocess.run([tmux, "has-session", "-t", session], capture_output=True).returncode != 0, "Queue already alive"
    if not args.resume_controller:
        assert not (RUNS / "queue_status.json").exists(), "Already started: use audited controller resume"
        assert subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip() == "feat/r18-c2-channelgate-author"
        assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip(), "Commit reviewed sources first"
        for file in ["raw/preflight.json", "raw/smoke_verification.json", "raw/resume_verification.json"]:
            assert json.loads((REPORT / file).read_text())["status"] == "passed", file
        frozen = RUNS / "source";assert not frozen.exists()
        prefix = str(EXP.relative_to(REPO)) + "/"
        files = subprocess.check_output(["git", "ls-files", "-z", "--", str(EXP.relative_to(REPO))], cwd=REPO).decode().split("\0")
        manifest = {}
        for rel in filter(None, files):
            local = rel.removeprefix(prefix);target = frozen / local
            target.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(REPO / rel, target)
            manifest[local] = sha(target)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        write(RUNS / "source_identity.json", {"commit": commit, "branch": "feat/r18-c2-channelgate-author", "manifest": manifest})
        write(REPORT / "source_identity.json", {"commit": commit, "branch": "feat/r18-c2-channelgate-author",
            "snapshot": str(frozen), "manifest_sha256": sha(RUNS / "source_identity.json"), "tracked_source_files": len(manifest)})
    else:
        assert (RUNS / "source_identity.json").exists()
        assert json.loads((RUNS / "queue_status.json").read_text())["status"] != "complete"
        frozen = RUNS / "source"
    config = RUNS / "yolo_config";config.mkdir(exist_ok=True)
    font = __import__("pathlib").Path.home() / ".config/Ultralytics/Arial.ttf"
    assert font.exists();shutil.copyfile(font, config / "Arial.ttf")
    env = {"PYTHONPATH": f"{frozen / 'backend'}:{frozen}", "AUTHOR_GATE_REPO": str(REPO),
        "AUTHOR_GATE_MANIFEST_SHA": sha(RUNS / "source_identity.json"), "PYTHONFAULTHANDLER": "1",
        "YOLO_CONFIG_DIR": str(config), "YOLO_AUTOINSTALL": "false", "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1", "MPLBACKEND": "Agg",
        "WANDB_MODE": "disabled", "COMET_MODE": "DISABLED"}
    launcher = RUNS / "launch.sh"
    launcher.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(frozen)) + "\nexec " +
        shlex.join(["env", *[f"{k}={v}" for k,v in env.items()], sys.executable, "-u", str(frozen / "controller.py")]) +
        " > " + shlex.quote(str(RUNS / "controller.log")) + " 2>&1\n")
    subprocess.run([tmux, "new-session", "-d", "-s", session, "bash", str(launcher)], check=True)
    write(REPORT / "launch.json", {"started_utc": now(), "session": session, "run_root": str(RUNS),
        "screen": "C2/Scalar/Channel 150 epochs seed0, fresh paired init",
        "confirmation": "Only on preregistered pass: C2/Channel 350 epochs seeds0,1,2, all fresh paired init",
        "environment": env})
    print("STARTED", session, flush=True)


if __name__ == "__main__":main()
