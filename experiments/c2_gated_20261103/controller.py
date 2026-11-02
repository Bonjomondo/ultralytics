"""Serialized, locked screen queue; runs only in the recorded frozen source tree."""
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from common import ARMS, EXP, REPO, REPORT, ROOT, RUNS, now, sha, write


def archive(arm):
    src, dst = RUNS / arm, REPORT / "raw" / arm
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["results.csv", "health.jsonl", "initialization.json", "args.yaml", "status.json"]:
        p = src / name
        if p.exists():
            shutil.copyfile(p, dst / name)
    for folder in [src, src / "formal_native"]:
        if folder.exists():
            for p in folder.glob("*.png"):
                if p.name.startswith(("results", "PR_curve", "P_curve", "R_curve", "F1_curve", "Box", "confusion_matrix")):
                    target = REPORT / "figures" / arm / folder.name / p.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, target)


def summarize():
    benchpath = REPORT / "raw/benchmark.json"
    bench = json.loads(benchpath.read_text())["models"] if benchpath.exists() else {}
    records = {a: json.loads(p.read_text()) for a in ARMS
               if (p := REPORT / f"raw/{a}/formal_metrics.json").exists()}
    rows = []
    for arm, rec in records.items():
        row = {"arm": arm, **rec["coco_percent"], "best_epoch": rec["best_epoch"]}
        if "c2" in records:
            row.update({f"delta_{k}_pp": rec["coco_percent"][k] - records["c2"]["coco_percent"][k]
                        for k in ["AP", "AP50", "AP75", "APs", "APm", "APl"]})
        if arm in bench:
            assert rec["checkpoint_sha256"] == bench[arm]["checkpoint_sha256"]
            row.update({k: bench[arm][k] for k in ["parameters", "gflops", "mean_ms", "median_ms", "p95_ms"]})
            row["latency_change_percent"] = 100 * (bench[arm]["median_ms"] / bench["c2"]["median_ms"] - 1)
            row["screen_pass"] = bool(arm != "c2" and row["delta_AP_pp"] >= .3 and
                row["delta_APs_pp"] >= .3 and row["latency_change_percent"] <= 10 and
                all(row["delta_" + k + "_pp"] >= -.5 for k in ["AP75", "APm", "APl"]))
        rows.append(row)
    if rows:
        with (REPORT / "comparison.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = ["# C2门控与融合筛选结果", "", "仅单seed、150轮筛选；原生mAP与独立COCO AP分别记录。",
             "主表为548图converted VisDrone、FP32、maxDets100；不等于VisDrone官方挑战赛指标。", "",
             "| arm | AP | APs | AP75 | ΔAP | ΔAPs | 前向中位ms | 晋级 |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['AP']:.3f} | {r['APs']:.3f} | {r['AP75']:.3f} | "
                     f"{r.get('delta_AP_pp', 0):+.3f} | {r.get('delta_APs_pp', 0):+.3f} | "
                     f"{r.get('median_ms', float('nan')):.3f} | {r.get('screen_pass', '待同场测速')} |")
    lines += ["", "晋级门槛：ΔAP与ΔAPs均≥0.3pp；AP75/APm/APl各下降不超过0.5pp；前向中位耗时增加≤10%。",
              "这是预设工程筛选门槛，不代表统计显著性或Orin部署性能。", "",
              "下一步：有晋级项再做同预算C2/候选350轮及多seed；本队列不自动追加或组合未经验证的模块。",
              "负面结果全部保留。新结果不能直接与历史另一引擎的28.72 AP比较。", ""]
    (REPORT / "RESULTS.md").write_text("\n".join(lines))


def commit_evidence(label):
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip() != "feat/r18-c2-gated-fusion":
        raise RuntimeError("Branch changed; evidence retained locally, automatic commit refused")
    if subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=REPO, text=True).strip():
        raise RuntimeError("Unrelated staged work exists; evidence retained without committing")
    relative = str(REPORT.relative_to(REPO))
    subprocess.run(["git", "add", "--", relative], cwd=REPO, check=True)
    subprocess.run(["git", "diff", "--cached", "--check"], cwd=REPO, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode:
        subprocess.run(["git", "commit", "-m", label], cwd=REPO, check=True)
    try:
        result = subprocess.run(["git", "push", "origin", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=60)
        receipt = {"returncode": result.returncode, "output": (result.stdout + result.stderr)[-1500:]}
    except subprocess.TimeoutExpired:
        receipt = {"returncode": None, "error": "push timeout after 60s; local commit preserved"}
    write(RUNS / "push_receipt.json", {**receipt, "updated_utc": now()})


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    with (RUNS / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert os.environ.get("C2_GATED_REPO_ROOT"), "Run from launcher in frozen source tree"
        manifest = json.loads((RUNS / "source_manifest.json").read_text())
        state = {"status": "running", "pid": os.getpid(), "started_utc": now(), "stages": []}

        def save():
            state["updated_utc"] = now(); write(RUNS / "queue_status.json", state)

        def stage(arm, phase, cmd):
            assert all(sha(ROOT / k) == v for k, v in manifest.items()), "Frozen source changed"
            item = {"arm": arm, "phase": phase, "status": "running", "started_utc": now()}
            state["stages"].append(item); state.update(arm=arm, phase=phase); save()
            log = RUNS / "logs" / f"{arm}-{phase}.log"; log.parent.mkdir(exist_ok=True)
            with log.open("w") as f:
                child = subprocess.Popen(cmd, cwd=ROOT, env=os.environ.copy(), stdout=f, stderr=subprocess.STDOUT)
                state["child_pid"] = child.pid; save(); code = child.wait()
            item.update(returncode=code, finished_utc=now(), status="complete" if code == 0 else "failed")
            state["child_pid"] = None; save()
            if code:
                raise RuntimeError(f"{arm}/{phase} failed with {code}; see {log}")

        try:
            save()
            for arm in ARMS:
                run = RUNS / arm
                if (run / "weights/last.pt").exists():
                    raise RuntimeError(f"Existing weights at {run}; no implicit restart or duplicated training")
                stage(arm, "train", [sys.executable, "-u", str(EXP / "train.py"), "--arm", arm])
                for phase in ["predict", "coco"]:
                    stage(arm, phase, [sys.executable, "-u", str(EXP / "evaluate.py"), phase, "--arm", arm])
                status = json.loads((run / "status.json").read_text())
                status.update(status="complete", updated_utc=now()); write(run / "status.json", status)
                archive(arm); summarize()
                commit_evidence(f"Archive 150-epoch C2 fusion screen: {arm}")
            stage("all", "benchmark", [sys.executable, "-u", str(EXP / "evaluate.py"), "benchmark"])
            summarize()
            state.update(status="complete", finished_utc=now()); save()
            write(REPORT / "raw/final_queue_status.json", state)
            commit_evidence("Complete paired C2 gated fusion screening and same-session benchmarks")
        except BaseException:
            state.update(status="failed", error=traceback.format_exc(), updated_utc=now()); save()
            for arm in ARMS:
                if (RUNS / arm).exists():
                    archive(arm)
            write(REPORT / "raw/failure.json", state)
            raise


if __name__ == "__main__":
    main()
