"""Run the three-arm screen, then conditionally six paired 350-epoch confirmations."""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

from common import ARMS, EXP, REPO, REPORT, RUNS, dataset_identity, now, run_name, sha, write
from report import analyze, archive

BRANCH = "feat/r18-c2-channelgate-author"


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    with (RUNS / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = json.loads((RUNS / "source_identity.json").read_text())
        manifest = identity["manifest"]
        assert os.environ.get("AUTHOR_GATE_MANIFEST_SHA") == sha(RUNS / "source_identity.json")
        path = RUNS / "queue_status.json"
        state = json.loads(path.read_text()) if path.exists() else {"started_utc": now(), "events": [], "done": []}
        if state.get("status") == "complete":return
        state.update(status="running", pid=os.getpid(), updated_utc=now())
        write(path, state)

        def verify():
            assert all(sha(EXP / k) == v for k, v in manifest.items()), "Frozen source changed"

        def event(label, cmd, train=False):
            if label in state["done"]:return
            budget = 3 if not train else 2
            for attempt in range(1, budget + 1):
                verify()
                args = list(cmd)
                if train:
                    assert dataset_identity() == json.loads((REPORT / "dataset_identity.json").read_text()), "Dataset changed"
                    run = RUNS / label.split("/")[0]
                    last = run / "weights/last.pt"
                    if last.exists():
                        # Avoid duplicate completion if the parent died after the train process exited.
                        status = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
                        if status.get("status") in {"training_complete_pending_evaluation", "complete"}:
                            state["done"].append(label);write(path, state);return
                        # Preserve last/best and receipts before any explicit raw-state recovery.
                        saved = RUNS / "recovery_snapshots" / run.name / f"{int(time.time())}"
                        saved.mkdir(parents=True)
                        for p in [last, run / "weights/best.pt", run / "status.json", run / "failure.json"]:
                            if p.exists():shutil.copyfile(p, saved / p.name)
                        args.append("--resume")
                    elif (run / "results.csv").exists():
                        raise RuntimeError("Curve exists without durable checkpoint: inspect rather than restarting")
                item = {"label": label, "attempt": attempt, "started_utc": now(), "status": "running"}
                state["events"].append(item);state.update(current=label, updated_utc=now())
                logfile = RUNS / "logs" / f"{label.replace('/', '-')}-attempt{len(state['events'])}.log"
                logfile.parent.mkdir(exist_ok=True)
                env = os.environ.copy()
                if "/coco" in label:env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                with logfile.open("w") as output:
                    child = subprocess.Popen([sys.executable, "-u", *args], cwd=EXP, env=env, stdout=output, stderr=subprocess.STDOUT)
                    state["child_pid"] = child.pid;write(path, state)
                    while child.poll() is None:
                        time.sleep(15)
                        # No tool thread is blocked here: this runs detached on the server.
                        if train:
                            statusfile = run / "status.json"
                            age = time.time() - max(logfile.stat().st_mtime, statusfile.stat().st_mtime if statusfile.exists() else 0)
                            if age > 1800:
                                child.terminate();item["stale_process_terminated"] = True
                                try:child.wait(timeout=30)
                                except subprocess.TimeoutExpired:child.kill()
                    code = child.wait()
                item.update(status="complete" if code == 0 else "failed", returncode=code,
                            finished_utc=now(), log=str(logfile))
                state.update(child_pid=None, updated_utc=now());write(path, state)
                if code == 0:
                    state["done"].append(label);write(path, state);return
                if train and not last.exists():
                    raise RuntimeError(f"{label} failed before a durable checkpoint; see {logfile}")
            raise RuntimeError(f"{label} failed after {budget} attempts; no further automatic retry")

        def publish(label):
            if subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip() != BRANCH:
                write(RUNS / "publication_pending.json", {"reason": "Repository branch changed", "updated_utc": now()});return
            if subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=REPO, text=True).strip():
                write(RUNS / "publication_pending.json", {"reason": "Other staged work present", "updated_utc": now()});return
            members = sorted(p for p in REPORT.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
            (REPORT / "MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(REPORT)}\n" for p in members))
            subprocess.run(["git", "add", "--", str(REPORT.relative_to(REPO))], cwd=REPO, check=True)
            for p in members:
                subprocess.run(["git", "ls-files", "--error-unmatch", str(p.relative_to(REPO))], cwd=REPO, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "diff", "--cached", "--check"], cwd=REPO, check=True)
            if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode:
                subprocess.run(["git", "commit", "-m", label], cwd=REPO, check=True)
            # The server has no GitHub credentials. A local heartbeat can push these commits.
            write(RUNS / "publication_pending.json", {"branch": BRANCH,
                  "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "updated_utc": now()})

        def run_one(arm, epochs, seed):
            name = run_name(arm, seed, epochs)
            if f"{name}/archived" in state["done"]:return
            event(f"{name}/train", [str(EXP / "train.py"), "--arm", arm, "--epochs", str(epochs), "--seed", str(seed)], True)
            for phase in ["predict", "coco"]:
                event(f"{name}/{phase}", [str(EXP / "evaluate.py"), phase, "--run", name])
            run = RUNS / name
            rows = __import__("csv").DictReader((run / "results.csv").open())
            assert len(list(rows)) == epochs, "Complete fixed budget is required"
            status = json.loads((run / "status.json").read_text());status.update(status="complete")
            write(run / "status.json", status);archive(name)
            state["done"].append(f"{name}/archived");write(path, state)
            # All model weights and eval predictions must agree, not just the final table.
            m = json.loads((run / "evaluation/metrics.json").read_text())
            assert sha(run / "weights/best.pt") == m["checkpoint_sha256"]
            analyze();publish(f"Archive author-engine paired gate run: {name}")

        try:
            for arm in ARMS:run_one(arm, 150, 0)
            event("screen/benchmark", [str(EXP / "evaluate.py"), "benchmark", "--models",
                *[run_name(a) for a in ARMS], "--output", "benchmark_screen.json"])
            shutil.copyfile(RUNS / "benchmark_screen.json", REPORT / "benchmark_screen.json")
            rows = analyze()
            base = next(r for r in rows if r["run"] == run_name("c2"))
            scalar = next(r for r in rows if r["run"] == run_name("scalar"))
            channel = next(r for r in rows if r["run"] == run_name("channel"))
            # Verify actual paired initialization across all complete training runs.
            receipts = [json.loads((RUNS / run_name(a) / "initialization.json").read_text()) for a in ARMS]
            assert len({r["shared_sha256"] for r in receipts}) == 1
            checks = {"AP_gain_at_least_0_3pp": channel["AP"] - base["AP"] >= .3,
                "APs_gain_at_least_0_3pp": channel["APs"] - base["APs"] >= .3,
                "AP75_APm_APl_no_drop_beyond_0_5pp": all(channel[k] - base[k] >= -.5 for k in ["AP75", "APm", "APl"]),
                "latency_increase_at_most_10percent": channel["median_ms"] <= base["median_ms"] * 1.1}
            passed = all(checks.values())
            write(REPORT / "decision.json", {"passed": passed, "checks": checks,
                "channel_minus_scalar_pp": {k: channel[k] - scalar[k] for k in ["AP", "APs", "AP75", "APm", "APl"]},
                "next": "fresh 350e C2/channel paired seeds0,1,2" if passed else "screen complete; preserve negative results; no automatic 350e",
                "caveat": "Engineering screening, not statistical significance. Historical C2 AP28.722 is context only."})
            analyze();publish("Complete three-arm author gate screen and preregistered decision")
            if passed:
                for seed in [0, 1, 2]:
                    for arm in ["c2", "channel"]:run_one(arm, 350, seed)
                    event(f"confirm_s{seed}/benchmark", [str(EXP / "evaluate.py"), "benchmark", "--models",
                        *[run_name(a, seed, 350) for a in ["c2", "channel"]], "--output", f"benchmark_confirm_s{seed}.json"])
                    shutil.copyfile(RUNS / f"benchmark_confirm_s{seed}.json", REPORT / f"benchmark_confirm_s{seed}.json")
                    analyze();publish(f"Archive same-session paired 350e benchmark seed{seed}")
            state.update(status="complete", finished_utc=now(), child_pid=None, confirmation_triggered=passed)
            write(path, state);write(REPORT / "final_queue_status.json", state)
            analyze();publish("Finalize author channel-gate experiment and complete evidence")
        except BaseException:
            state.update(status="failed", error=traceback.format_exc(), updated_utc=now());write(path, state)
            write(REPORT / "failure.json", state)
            # Keep a readable snapshot of any completed epochs and diagnostic values.
            for run in RUNS.glob("screen_*_150e_s0"):
                if run.is_dir():archive(run.name)
            raise


if __name__ == "__main__":main()
