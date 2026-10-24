"""Run seed0 C2/C0 to an empirical plateau, evaluate and archive matched-budget checkpoints."""
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiments.r18_p2_sfi.continue_train import plateau_status

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/r18_p2_sfi"
OUT = EXP / "convergence_20261024"
RUNS = ROOT / "runs/detect/r18_p2_sfi_convergence_20261024"
REPORT = ROOT / "docs/experiment_reports/2026-10-24_r18_p2_sfi_convergence"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def source_manifest():
    paths = [*sorted((ROOT / "ultralytics").rglob("*.py")),
             *sorted((ROOT / "ultralytics/cfg").rglob("*.yaml")), *sorted(EXP.glob("*.py")),
             ROOT / "train_rtdetr_r18_na.py", ROOT / "train_paper_repro_visdrone.py",
             ROOT / "configs/visdrone-local.yaml"]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def make_report(state):
    REPORT.mkdir(parents=True, exist_ok=True)
    raw = REPORT / "raw"
    raw.mkdir(exist_ok=True)
    lines = ["# C0/C2 seed0 分阶段收敛对比", "", f"状态：{state['status']}。",
             "", "[150轮核查、配置与初始化证据](../2026-10-24_r18_p2_sfi_audit/README.md)。",
             "原checkpoint优化器已清理；本阶段从第150轮FP16 EMA warm start、重建AdamW/EMA、相同3轮预热；不是连续从零训练。",
             "经验平台判据：追加>=150轮，50轮最高内部mAP增益<=0.1pp，且相邻20轮均值差绝对值<=0.1pp；追加350轮为上限，触顶不视为收敛。",
             "", "## 各组最终best（预算可不同）", "",
             "| 组别 | 追加轮数 | 累计轮数 | 经验平台 | 正式AP (%) | APs (%) | 相对自身150轮ΔAP (pp) |",
             "| --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for c in state["completed_training"]:
        run = RUNS / f"rtdetr-r18-{c}-seed0-cont"
        for name, target in [("results.csv", f"{c}_training_curve.csv"), ("args.yaml", f"{c}_effective_args.yaml"),
                             ("continuation.json", f"{c}_continuation.json"),
                             ("convergence_status.json", f"{c}_convergence_status.json")]:
            shutil.copy2(run / name, raw / target)
        if c not in state["completed_evaluation"]:
            continue
        d = json.loads((OUT / "final" / f"{c}_formal_metrics.json").read_text())
        initial = json.loads((EXP / "corrected_20261024" / f"{c}_formal_metrics.json").read_text())
        plateau = json.loads((run / "convergence_status.json").read_text())
        lines.append(f"| {c} | {d['training_epochs']} | {150+d['training_epochs']} | {plateau['converged']} | {100*d['metrics']['AP']:.3f} | {100*d['metrics']['APs']:.3f} | {100*(d['metrics']['AP']-initial['metrics']['AP']):+.3f} |")
    lines.extend(["", "各阶段best按内部mAP选择，正式COCOeval仅用于报告；保留原150轮结果以防追加训练退化。",
                  "", "## 等预算对比", "", "下面每一行在两组相同追加训练预算内选择best，避免把不同停止轮数直接当结构收益。",
                  "| 累计预算 | C0 AP (%) | C2 AP (%) | C2-C0 AP (pp) | C2-C0 APs (pp) |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for budget in state.get("matched_budgets", []):
        folder = OUT / f"matched_additional{budget}"
        a, b = [json.loads((folder / f"{c}_formal_metrics.json").read_text()) for c in ("c0", "c2")]
        lines.append(f"| {150+budget} | {100*a['metrics']['AP']:.3f} | {100*b['metrics']['AP']:.3f} | {100*(b['metrics']['AP']-a['metrics']['AP']):+.3f} | {100*(b['metrics']['APs']-a['metrics']['APs']):+.3f} |")
    lines.extend(["", "原始权重、逐图预测、完整日志及可恢复resume.pt仅留服务器；正式指标JSON与全曲线纳入仓库。",
                  "[阶段正式指标](../../../experiments/r18_p2_sfi/convergence_20261024/final/RESULTS.md)", ""])
    # Keep the report valid before the final-results directory exists.
    if not (OUT / "final/RESULTS.md").exists():
        lines = [line for line in lines if not line.startswith("[阶段正式指标]")]
    (REPORT / "README.md").write_text("\n".join(lines))
    members = sorted(p for p in REPORT.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    (REPORT / "MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(REPORT)}\n" for p in members))


def main():
    os.chdir(ROOT)
    os.environ.update(PYTHONPATH=str(ROOT), MPLBACKEND="Agg", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4")
    lock = Path("/tmp/r18-p2-sfi-gpu0.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    OUT.mkdir(exist_ok=True)
    status_path = OUT / "status.json"
    if status_path.exists():
        raise FileExistsError("Existing queue state; refusing duplicate launch")
    if json.loads((EXP / "corrected_20261024/status.json").read_text())["status"] != "completed":
        raise RuntimeError("Fixed-mask reevaluation must complete first")
    c0, c2 = [json.loads((EXP / "corrected_20261024" / f"{c}_formal_metrics.json").read_text()) for c in ("c0", "c2")]
    if (c2["metrics"]["AP"]-c0["metrics"]["AP"])*100 < 0.3 or (c2["metrics"]["APs"]-c0["metrics"]["APs"])*100 < 0.5:
        raise RuntimeError("C2 no longer clears the initial accuracy screen")
    frozen = source_manifest()
    write(OUT / "source_sha256.json", frozen)
    plan = {"order": ["c2", "c0"], "seed": 0, "source_epochs": 150, "minimum_additional_epochs": 150,
            "maximum_additional_epochs": 350, "optimizer_restart": True, "lr0": 0.0001, "lrf": 1.0,
            "warmup_epochs": 3, "batch": 8, "nbs": 64, "amp": False,
            "plateau": "50-epoch best gain<=0.1pp and abs(mean20-prev20)<=0.1pp after >=150 additional epochs",
            "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_last_sha256": {c:sha(ROOT / f"runs/detect/r18_p2_sfi_20261023/rtdetr-r18-{c}-150e-s0/weights/last.pt") for c in ("c0","c2")}}
    write(OUT / "plan.json", plan)
    state = {"status": "running", "started_utc": now(), "pid": os.getpid(),
             "completed_training": [], "completed_evaluation": [], "matched_budgets": [], "phase": "starting"}

    def run(command, label):
        if source_manifest() != frozen:
            raise RuntimeError("Frozen experiment sources changed")
        log = ROOT / f"logs/p2-sfi-convergence-{label}-20261024.log"
        state.update(command=command, log=str(log), updated_utc=now())
        write(status_path, state)
        with log.open("x") as file:
            child = subprocess.Popen(command, stdout=file, stderr=subprocess.STDOUT)
            state["child_pid"] = child.pid
            write(status_path, state)
            rc = child.wait()
        if rc:
            raise RuntimeError(f"{label} exited {rc}; inspect {log}")

    try:
        for c in plan["order"]:
            state.update(variant=c, phase="training")
            run_dir = RUNS / f"rtdetr-r18-{c}-seed0-cont"
            if run_dir.exists():
                raise FileExistsError(f"Partial run exists: {run_dir}; resume explicitly")
            run([sys.executable, "-u", str(EXP / "continue_train.py"), "--variant", c], f"{c}-training")
            rows = list(csv.DictReader((run_dir / "results.csv").open()))
            progress = plateau_status([float(row["metrics/mAP50-95(B)"]) for row in rows])
            if len(rows) < 350 and not progress["converged"]:
                raise RuntimeError("Training stopped before plateau or budget cap")
            state["completed_training"].append(c)
            state["phase"] = "evaluation"
            run([sys.executable, "-u", str(EXP / "evaluate.py"), "--variant", c, "--run", str(run_dir),
                 "--expected-epochs", "350", "--allow-partial", "--output-dir", str(OUT / "final")], f"{c}-final")
            state["completed_evaluation"].append(c)
            write(status_path, state)
            make_report(state)
        lengths = [len(list(csv.DictReader((RUNS / f"rtdetr-r18-{c}-seed0-cont/results.csv").open()))) for c in ("c0","c2")]
        for budget in range(50, min(lengths)+1, 50):
            for c in ("c0", "c2"):
                state.update(variant=c, phase=f"matched_budget_{budget}")
                run_dir = RUNS / f"rtdetr-r18-{c}-seed0-cont"
                run([sys.executable, "-u", str(EXP / "evaluate.py"), "--variant", c, "--run", str(run_dir),
                     "--expected-epochs", "350", "--allow-partial", "--checkpoint",
                     str(run_dir / f"weights/additional{budget}_best.pt"),
                     "--output-dir", str(OUT / f"matched_additional{budget}")], f"{c}-matched{budget}")
                # Evaluation budget identifies checkpoint selection horizon rather than final CSV length.
                record_path = OUT / f"matched_additional{budget}/{c}_formal_metrics.json"
                record = json.loads(record_path.read_text())
                record.update(training_epochs=budget, available_curve_epochs=lengths[0 if c == "c0" else 1],
                              checkpoint_selection_horizon_additional_epochs=budget)
                write(record_path, record)
            state["matched_budgets"].append(budget)
            make_report(state)
        # Regenerate matched summaries after explicit checkpoint-horizon annotation.
        import experiments.r18_p2_sfi.evaluate as evaluation
        for budget in state["matched_budgets"]:
            evaluation.OUTPUT = OUT / f"matched_additional{budget}"
            evaluation.summarize()
        state.update(status="completed", phase="done", child_pid=None, finished_utc=now())
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", failed_utc=now())
        raise
    finally:
        write(status_path, state)
        make_report(state)


if __name__ == "__main__":
    main()
