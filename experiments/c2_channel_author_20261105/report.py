"""Archive compact evidence and rebuild plots/tables without the GPU or server weights."""
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from common import REPORT, RUNS, now, sha, write


def archive(name):
    run, dst = RUNS / name, REPORT / "raw" / name
    dst.mkdir(parents=True, exist_ok=True)
    for filename in ["results.csv", "args.yaml", "initialization.json", "first_batch.json", "status.json"]:
        if (run / filename).exists():shutil.copyfile(run / filename, dst / filename)
    for filename in ["metrics.json", "prediction_metadata.json"]:
        if (run / "evaluation" / filename).exists():shutil.copyfile(run / "evaluation" / filename, dst / filename)
    if (run / "diagnostics.jsonl").exists():
        diagnostics = [json.loads(s) for s in (run / "diagnostics.jsonl").read_text().splitlines()]
        with (dst / "alpha.csv").open("w", newline="") as f:
            w = csv.writer(f);w.writerow(["epoch", *[f"alpha_{i:03d}" for i in range(128)]])
            for d in diagnostics:
                a = d["alpha"] if len(d["alpha"]) == 128 else d["alpha"] * 128
                w.writerow([d["epoch"], *a])
        rows = []
        for d in diagnostics:
            row = {k: d[k] for k in ["epoch", "mean", "std", "min", "max", "below_0_9", "above_1_1", "low_saturation", "high_saturation", "epoch_seconds", "peak_gpu_gib"]}
            for fields, key in [(["p2_rms_before", "p2_rms_after"], "p2_feature_rms_before_after"),
                                (["p2_gradient_norm", "gate_gradient_norm"], "p2_gate_grad_norm")]:
                row.update(dict(zip(fields, d[key])))
            rows.append(row)
        if rows:
            with (dst / "diagnostics.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    figures = {}
    for folder in [run, run / "evaluation"]:
        for p in folder.glob("*.png"):
            if p.name not in ["results.png", "F1_curve.png", "P_curve.png", "R_curve.png", "PR_curve.png", "confusion_matrix.png", "confusion_matrix_normalized.png"]:continue
            target = REPORT / "figures" / name / p.name
            target.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(p, target)
            figures[str(target.relative_to(REPORT))] = sha(target)
    if (run / "results.csv").exists():assert sha(run / "results.csv") == sha(dst / "results.csv")
    write(dst / "server_evidence.json", {"run": str(run), "full_diagnostics_sha256": sha(run / "diagnostics.jsonl") if (run / "diagnostics.jsonl").exists() else None,
        "sampling": "features/gradients first 3 batches per epoch; alpha at epoch end; scalar/C2 broadcast only in alpha.csv",
        "figure_hashes": figures, "native_figure_protocol": "framework internal validation; not independent COCO AP"})


def analyze(report=REPORT):
    """Only consume files in the Git archive; safe to run after checkout elsewhere."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in fonts), "sans-serif")
    plt.rcParams.update({"font.family": family,
        "font.size": 14, "axes.linewidth": 1.8, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    rows = []
    for path in sorted((report / "raw").glob("*/metrics.json")):
        d = json.loads(path.read_text());name = path.parent.name
        if name.startswith("smoke"):continue
        stage, arm, budget, seed = name.split("_")
        row = {"run": name, "stage": stage, "arm": arm, "budget": int(budget[:-1]), "seed": int(seed[1:]),
               **d["coco_percent"], "best_epoch": d["checkpoint_epoch"], "checkpoint_sha256": d["checkpoint_sha256"]}
        benchfile = report / ("benchmark_screen.json" if stage == "screen" else f"benchmark_confirm_s{row['seed']}.json")
        if benchfile.exists():
            bench = json.loads(benchfile.read_text())["models"].get(name)
            if bench:
                assert bench["checkpoint_sha256"] == d["checkpoint_sha256"]
                row.update({k: bench[k] for k in ["parameters", "median_ms", "p95_ms"]})
        rows.append(row)
    for row in rows:
        base = next((r for r in rows if r["stage"] == row["stage"] and r["seed"] == row["seed"] and r["arm"] == "c2"), None)
        if base:
            for k in ["AP", "AP50", "AP75", "APs", "APm", "APl"]:row[f"delta_{k}_pp"] = row[k] - base[k]
            if "median_ms" in row and "median_ms" in base:row["latency_change_percent"] = 100 * (row["median_ms"] / base["median_ms"] - 1)
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (report / "comparison.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields);w.writeheader();w.writerows(rows)
    lines = ["# Author-engine C2门控配对实验", "", "主表：FP32、converted VisDrone val548、COCO maxDets100。", "",
             "| run | AP | APs | AP75 | ΔAP | ΔAPs | GPU中位ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        latency = f"{r['median_ms']:.3f}" if "median_ms" in r else "待同场测速"
        lines.append(f"| {r['run']} | {r['AP']:.3f} | {r['APs']:.3f} | {r['AP75']:.3f} | {r.get('delta_AP_pp',0):+.3f} | {r.get('delta_APs_pp',0):+.3f} | {latency} |")
    decision = report / "decision.json"
    if decision.exists():lines.extend(["", "筛选决定：", "", "```json", decision.read_text().strip(), "```"])
    confirmation = [r for r in rows if r["stage"] == "confirm"]
    summary = {}
    for arm in ["c2", "channel"]:
        values = [r for r in confirmation if r["arm"] == arm]
        if len(values) == 3:
            summary[arm] = {k: {"mean": float(np.mean([r[k] for r in values])), "sample_std": float(np.std([r[k] for r in values], ddof=1))} for k in ["AP", "APs", "AP75", "APm", "APl"]}
    if len(summary) == 2:
        write(report / "multi_seed_summary.json", summary)
        lines += ["", "三seed均值与样本标准差见multi_seed_summary.json。n=3不等于证明统计显著性；配对差值另见comparison.csv。"]
    lines += ["", "历史28.722 AP不是本轮唯一基线；本轮所有比较使用同批次新训练的配对C2。", "所有负面结果均保留。150轮筛选不等于证明收敛。"]
    (report / "RESULTS.md").write_text("\n".join(lines) + "\n")
    for path in (report / "raw").glob("*/alpha.csv"):
        name = path.parent.name
        if name.startswith("smoke"):continue
        a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        if not len(a):continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        epochs, v = a[:, 0], a[:, 1:]
        axes[0].plot(epochs, v.mean(1), color="#0F4D92", lw=2, marker="o" if len(epochs)<3 else None, label="Channel mean")
        axes[0].fill_between(epochs, v.min(1), v.max(1), color="#42949E", alpha=.25, label="Channel min-max")
        axes[0].axhline(1, color="#767676", ls="--", lw=1)
        axes[0].set_ylim(min(.95, float(v.min())-.02), max(1.05, float(v.max())+.02))
        axes[0].set_xlim(epochs[0]-.5, epochs[-1]+.5)
        axes[0].ticklabel_format(axis="y", useOffset=False)
        axes[0].set(xlabel="Epoch", ylabel="Alpha", title="Static gate parameters");axes[0].legend(fontsize=10)
        im = axes[1].imshow(v.T, aspect="auto", origin="lower", extent=[epochs[0]-.5, epochs[-1]+.5, -.5,127.5], cmap="coolwarm", vmin=0, vmax=2)
        axes[1].set(xlabel="Epoch", ylabel="P2 channel", title="All 128 channels")
        fig.colorbar(im, ax=axes[1], label="Alpha");fig.tight_layout(pad=1.3)
        out = report / "figures" / name;out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "alpha_history.png", dpi=300);fig.savefig(out / "alpha_history.svg");plt.close(fig)
    return rows


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser();p.add_argument("--report", type=Path, default=REPORT);args = p.parse_args();analyze(args.report)
