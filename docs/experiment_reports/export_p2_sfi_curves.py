"""Archive original P2-SFI plots and training CSV snapshots without changing frozen training sources."""
import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/experiment_reports/2026-10-24_r18_p2_sfi_curves"
PLOTS = ("BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png", "BoxF1_curve.png",
         "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png")
STATUS = ROOT / "experiments/r18_p2_sfi/convergence_20261024/status.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != data:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)


def collect():
    entries = []
    state = json.loads(STATUS.read_text()) if STATUS.exists() else {"status": "not_started"}
    runs = [(c, ROOT / f"runs/detect/r18_p2_sfi_20261023/rtdetr-r18-{c}-150e-s0", False)
            for c in ("c0", "c1", "c2", "c3")]
    runs += [(f"{c}_continuation", ROOT / f"runs/detect/r18_p2_sfi_convergence_20261024/rtdetr-r18-{c}-seed0-cont", True)
             for c in ("c2", "c0")]
    for name, run, continuation in runs:
        path = run / "results.csv"
        if not path.exists():
            continue
        data = path.read_bytes()
        if not data.endswith(b"\n"):
            continue  # a write may be in progress; retry next collection
        rows = list(csv.DictReader(data.decode().splitlines()))
        if not rows or len(rows) != int(rows[-1]["epoch"]):
            continue
        if continuation:
            target = OUT / f"raw/{name}/results.csv"
            save(target, data)
        else:
            # Reuse the already-tracked byte-identical CSV, without a duplicate copy.
            target = ROOT / f"docs/experiment_reports/2026-10-24_r18_p2_sfi_audit/raw/{name}_training_curve.csv"
            if target.read_bytes() != data:
                raise RuntimeError(f"Archived 150-epoch curve differs from source: {name}")
        plots = []
        for filename in PLOTS:
            src = run / filename
            if src.exists():
                image = src.read_bytes()
                if not image.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise RuntimeError(f"Invalid PNG: {src}")
                dest = OUT / f"figures/{name}/{filename}"
                save(dest, image)
                plots.append({"path": str(dest.relative_to(OUT)), "sha256": digest(image), "bytes": len(image)})
        entries.append({"run": name, "server_source": str(run), "epochs": len(rows),
                        "cumulative_epochs": len(rows) + (150 if continuation else 0),
                        "csv": os.path.relpath(target, OUT), "csv_sha256": digest(data),
                        "snapshot": continuation and name[:2] not in state.get("completed_training", []), "plots": plots})
    lines = ["# P2-SFI 原始训练曲线与验证图表", "", "分支：`feat/r18-p2-sfi`。C0–C3为已完成150轮；接续训练仍在运行时，CSV明确标记为快照。",
             "", "CSV为训练目录results.csv的逐字节归档，150轮文件复用已有Git记录。PNG从原始训练产物直接复制，无重绘或压缩改写。",
             "PR/P/R/F1、混淆矩阵是框架原生验证口径，不能与正式COCOeval AP混用；results.png为各轮训练/验证指标。",
             "", "| 实验 | 阶段轮数 / 累计轮数 | CSV | PR | 训练曲线 | P / R / F1 | 混淆矩阵 | 状态 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for entry in entries:
        paths = {Path(p["path"]).name: p["path"] for p in entry["plots"]}
        def link(label, filename):
            return f"[{label}]({paths[filename]})" if filename in paths else "待生成"
        lines.append(f"| {entry['run']} | {entry['epochs']} / {entry['cumulative_epochs']} | [results.csv]({entry['csv']}) | "
                     f"{link('PR', 'BoxPR_curve.png')} | {link('results', 'results.png')} | "
                     f"{link('P','BoxP_curve.png')} / {link('R','BoxR_curve.png')} / {link('F1','BoxF1_curve.png')} | "
                     f"{link('原始','confusion_matrix.png')} / {link('归一化','confusion_matrix_normalized.png')} | "
                     f"{'运行中快照' if entry['snapshot'] else '已完成'} |")
    lines.extend(["", "[文件来源、轮次和SHA256](artifacts.json)。每次采集只保留一份当前快照，Git保存已提交历史。",
                  "原生PNG通常在训练结束时生成；运行中尚未生成的图表不以旧实验图替代。训练batch拼图、验证样本图、权重、数据集、逐图预测及完整日志只留服务器。",
                  "", "更新归档：`PYTHONPATH=. python docs/experiment_reports/export_p2_sfi_curves.py`。`--watch`每60秒更新一次，接续队列完成或失败后采集最后一次并退出。",
                  "自动采集只更新归档文件，不自动提交或推送；新增图表按本目录的Git例外规则纳入后续提交。", ""])
    save(OUT / "README.md", "\n".join(lines).encode())
    save(OUT / "artifacts.json", (json.dumps({"queue_status": state["status"], "runs": entries}, indent=2, ensure_ascii=False) + "\n").encode())
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256" and p.suffix != ".tmp")
    save(OUT / "MANIFEST.sha256", "".join(f"{digest(p.read_bytes())}  {p.relative_to(OUT)}\n" for p in files).encode())
    return state["status"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        status = collect()
        print(datetime.now(timezone.utc).isoformat(), "archive updated; queue", status, flush=True)
        if not args.watch or status in {"completed", "failed"}:
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
