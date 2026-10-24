"""Build compact, reproducible evidence and audit report from completed fixed-mask evaluations."""
import csv
import hashlib
import json
import platform
import shutil
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/r18_p2_sfi"
FIXED = EXP / "corrected_20261024"
REPORT = ROOT / "docs/experiment_reports/2026-10-24_r18_p2_sfi_audit"
RAW = REPORT / "raw"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    if "--collect" in sys.argv:
        records, summary, training = {}, [], []
        for c in ("c0", "c1", "c2", "c3"):
            run = ROOT / f"runs/detect/r18_p2_sfi_20261023/rtdetr-r18-{c}-150e-s0"
            new = json.loads((FIXED / f"{c}_formal_metrics.json").read_text())
            old = json.loads((EXP / f"{c}_formal_metrics.json").read_text())
            records[c] = new
            summary.append({"variant": c, "old_AP_pct": old["metrics"]["AP"] * 100,
                            "fixed_AP_pct": new["metrics"]["AP"] * 100,
                            "delta_AP_pp": (new["metrics"]["AP"] - old["metrics"]["AP"]) * 100,
                            "predictions_identical": old["predictions_file_sha256"] == new["predictions_file_sha256"],
                            **new["confidence_filter_audit"]})
            shutil.copy2(run / "results.csv", RAW / f"{c}_training_curve.csv")
            shutil.copy2(run / "initialization.json", RAW / f"{c}_initialization.json")
            write(RAW / f"{c}_effective_args.json", YAML.load(run / "args.yaml"))
            rows = list(csv.DictReader((run / "results.csv").open()))
            key = "metrics/mAP50-95(B)"
            best = max(rows, key=lambda row: float(row[key]))
            training.append({"variant": c, "epochs": len(rows), "best_internal_epoch": int(best["epoch"]),
                             "best_internal_mAP_pct": 100 * float(best[key]),
                             "mean_internal_mAP_epochs131_140_pct": 10 * sum(float(row[key]) for row in rows[130:140]),
                             "mean_internal_mAP_epochs141_150_pct": 10 * sum(float(row[key]) for row in rows[140:150]),
                             "run_path": str(run), "last_checkpoint_sha256": sha(run / "weights/last.pt"),
                             "best_checkpoint_sha256": new["checkpoint_sha256"]})
        write(RAW / "postprocess_comparison.json", summary)
        write(RAW / "training_summary.json", training)
        init = [json.loads((RAW / f"{c}_initialization.json").read_text()) for c in records]
        assert len({x["shared_state_sha256"] for x in init}) == 1
        assert all(x["shared_state_verified"] for x in init)
        gt_path = ROOT.parent / "datasets/VisDrone2019-DET-YOLO/annotations/VisDrone2019-DET_val_coco.json"
        gt = json.loads(gt_path.read_text())
        counts = []
        for category in gt["categories"]:
            anns = [a for a in gt["annotations"] if a["category_id"] == category["id"]]
            for scale, lo, hi in [("all", 0, 1e10), ("small", 0, 1024), ("medium", 1024, 9216), ("large", 9216, 1e10)]:
                selected = [a for a in anns if lo <= a["area"] <= hi]
                eligible = [a for a in selected if not a.get("iscrowd", 0) and not a.get("ignore", 0)]
                counts.append({"category": category["name"], "scale": scale, "gt_count": len(selected),
                               "noncrowd_nonignore_count": len(eligible),
                               "images_with_gt": len({a["image_id"] for a in selected})})
        with (RAW / "gt_category_scale_counts.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(counts[0])); writer.writeheader(); writer.writerows(counts)
        classes = []
        for category, values in records["c2"]["per_class"].items():
            classes.append({"category": category, **{f"delta_{key}_pp": 100 * (value - records["c0"]["per_class"][category][key])
                             for key, value in values.items()}})
        write(RAW / "c2_vs_c0_class_deltas.json", classes)
        write(RAW / "environment.json", {"python": platform.python_version(), "torch": torch.__version__,
              "gpu": records["c0"]["benchmark"]["gpu"], "GT_sha256": sha(gt_path),
              "base_commit": "38e574cc1ff112be35a6aca6516b5d7fecc25149", "branch": "feat/r18-p2-sfi",
              "generated_utc": datetime.now(timezone.utc).isoformat(), "cuda": torch.version.cuda})
        paths = [ROOT / "ultralytics/models/rtdetr/val.py", *EXP.glob("*.py")]
        write(RAW / "audit_source_sha256.json", {str(p.relative_to(ROOT)): sha(p) for p in paths})
    else:
        records = {c: json.loads((FIXED / f"{c}_formal_metrics.json").read_text()) for c in ("c0", "c1", "c2", "c3")}
        training = json.loads((RAW / "training_summary.json").read_text())
    lines = ["# P2-SFI 后处理核查与150轮结果", "", "2026-10-24，分支 `feat/r18-p2-sfi`；输入提交 `38e574cc`。",
             "", "后处理确有排序后仍使用旧顺序score掩码的问题，已改为对排序后的pred[:,4]做阈值过滤。",
             "四组各548图、164400个Query，在conf=0.001下均无Query被过滤；每组修复前后预测JSON SHA256完全相同，正式COCO指标也完全相同。",
             "因此该bug不影响本次四个已选best权重的验证结果。未保存的历史epoch无法追溯逐Query分数，不能据此宣称所有训练期间best选择均完全不受影响。",
             "", "[逐组核查摘要](raw/postprocess_comparison.json) · [修复后正式结果](../../../experiments/r18_p2_sfi/corrected_20261024/RESULTS.md) · [完整指标表](../../../experiments/r18_p2_sfi/corrected_20261024/formal_metrics_comparison.csv)",
             "", "| 组别 | 正式AP (%) | APs (%) | 内部best轮次 | 内部best mAP (%) | 最后两段10轮均值 (%) |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for c, row in zip(records, training):
        d = records[c]
        lines.append(f"| {c} | {100*d['metrics']['AP']:.3f} | {100*d['metrics']['APs']:.3f} | {row['best_internal_epoch']} | {row['best_internal_mAP_pct']:.3f} | {row['mean_internal_mAP_epochs131_140_pct']:.3f} → {row['mean_internal_mAP_epochs141_150_pct']:.3f} |")
    lines.extend(["", "四组末期仍上涨；150轮尚不能视为收敛。正式AP与内部mAP口径不同，不混用。C2保留+0.704pp AP、+0.829pp APs。",
                  "", "## 证据与边界", "", "- raw/ 中四组各包含完整150轮CSV曲线、实际配置JSON与原始初始化记录；共有初始化SHA已交叉校验一致。",
                  "- [训练摘要、best轮次和权重SHA](raw/training_summary.json)；权重及完整原始日志只留服务器。",
                  "- [类别×尺度GT数量与覆盖图数](raw/gt_category_scale_counts.csv)，尺度端点按COCO面积区间包含，1024/9216边界可重叠；另列非crowd/非ignore数量。",
                  "- [类别/尺度增量](raw/c2_vs_c0_class_deltas.json)：bus/van约占净APs增量84%；这是类别平均AP增量分解，不能解释为目标数量或图像贡献。尚未做逐图重采样归因。",
                  "- C0/C2同时改变P2接入与P3融合通道，机制仍需独立控制实验；本轮不新增结构、多seed或部署实验。",
                  "- 延迟为4090 batch1 fused PyTorch FP32 GPU forward，50次预热/200次计时，不含传输/预后处理，不代表Orin部署。",
                  "", "## 接续训练", "", "按用户要求先运行seed0的C2和C0至平台期。原150轮last.pt已被框架strip_optimizer，优化器状态和epoch被清除，权重为FP16保存的EMA。",
                  "本次从各自第150轮last权重重新建立AdamW/EMA，并进行相同3轮预热；属于分阶段warm start，不是无缝恢复或从零连续500轮。",
                  "其余配方保持一致：640、batch8/nbs64、FP32、AdamW、恒定1e-4、seed0、原增强。",
                  "每组至少追加150轮；之后若最近50轮最高内部mAP相比更早最高值增加≤0.1pp，且最近20轮均值与前20轮均值差绝对值≤0.1pp，标记经验平台期。最多追加350轮（累计预算上限500轮）；触顶不自动宣称收敛。",
                  "每50轮同时保留last和当时best；最终另存未清理优化器的resume.pt。若独立平台停止预算不同，报告必须列出轮数，并利用公共50轮检查点作等预算对比。",
                  "脚本：`experiments/r18_p2_sfi/continue_train.py`；队列：`convergence_queue.py`。结束自动正式评估并生成最终报告。",
                  "", "原始150轮报告和旧指标保留，修复后结果另存corrected_20261024；原始权重不会覆盖。", "", "复生成本页：PYTHONPATH=. python experiments/r18_p2_sfi/archive_audit.py（仅需Git内原始表格，无GPU/服务器数据）。--collect为服务器证据采集模式，依赖外部权重/GT。", ""])
    if (RAW / "verification.json").exists():
        lines.extend(["验证：[19项测试及真实数据接续短跑记录](raw/verification.json)。短跑仅检查管线，不参与精度比较。", ""])
    (REPORT / "README.md").write_text("\n".join(lines))
    index = ROOT / "docs/experiment_reports/README.md"
    text = index.read_text()
    entry = "- [2026-10-24：P2-SFI 后处理核查、完整150轮证据与收敛续训](2026-10-24_r18_p2_sfi_audit/README.md)\n"
    if entry not in text:
        index.write_text(text.replace("# 实验结果与报告\n", "# 实验结果与报告\n\n" + entry))
    old_entry = ROOT / "docs/experiment_reports/2026-10-23_r18_p2_sfi/README.md"
    prefix = "> 更新：四组150轮与正式评估已完成；[2026-10-24核查与完整训练证据](../2026-10-24_r18_p2_sfi_audit/README.md)。以下保留当时启动快照。\n\n"
    if not old_entry.read_text().startswith("> 更新"):
        old_entry.write_text(prefix + old_entry.read_text())
    members = sorted(p for p in REPORT.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    (REPORT / "MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(REPORT)}\n" for p in members))


if __name__ == "__main__":
    main()
