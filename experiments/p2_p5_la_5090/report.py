"""Build a compact text report from four completed evaluations and matched timings (no GPU)."""
import argparse
import csv
import json
from pathlib import Path

ARMS = ("c0", "c2", "la", "c2-la")


def validate(evaluations, benchmark):
    if benchmark["status"] != "completed" or benchmark["role"] != "trained_checkpoint_comparison":
        raise ValueError("Report needs timings from trained checkpoints")
    if set(evaluations) != set(ARMS) or set(benchmark["models"]) != set(ARMS):
        raise ValueError("All four factorial arms are required")
    reference = evaluations["c0"]
    for arm, item in evaluations.items():
        if (item["status"] != "completed" or item["variant"] != arm or item["scope"] != "full_val"
                or item["training_role"] != "fixed_budget_screen" or item["training_fraction"] != 1
                or item["protocol"] != reference["protocol"]
                or any(item[key] != reference[key] for key in ("training_recipe", "training_source_sha256", "data_yaml_sha256"))
                or item["initialization"]["shared_state_sha256"] != reference["initialization"]["shared_state_sha256"]
                or (item["seed"], item["training_epochs"]) != (reference["seed"], reference["training_epochs"])):
            raise ValueError("Only matched full-validation/full-training-budget evaluations can be compared")
        if benchmark["models"][arm]["identity"]["sha256"] != item["checkpoint"]["sha256"]:
            raise ValueError("Timing checkpoint differs from accuracy checkpoint: " + arm)
    for slot, arms in (("p2_slot_sha256", ("c2", "c2-la")), ("p5_slot_sha256", ("la", "c2-la"))):
        if evaluations[arms[0]]["initialization"][slot] != evaluations[arms[1]]["initialization"][slot]:
            raise ValueError("Factorial branch initialization differs: " + slot)


def report(evaluations, benchmark, output):
    validate(evaluations, benchmark)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for arm in ARMS:
        accuracy, timing = evaluations[arm], benchmark["models"][arm]
        rows.append({"arm": arm, "seed": accuracy["seed"], "epochs": accuracy["training_epochs"],
                     **{key + "_percent": 100 * accuracy["metrics"][key] for key in ("AP", "AP50", "APs", "APm", "APl")},
                     "whole_ms": timing["summary"]["bare"]["whole"]["mean_ms"],
                     **{key + "_ms": timing["summary"]["staged"][key]["mean_ms"] for key in ("p5", "neck_encoder")},
                     "checkpoint_sha256": accuracy["checkpoint"]["sha256"]})
    with (output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_arm = {r["arm"]: r for r in rows}
    effects = {}
    for metric in ("AP_percent", "APs_percent", "whole_ms", "p5_ms", "neck_encoder_ms"):
        a, b, c, d = [by_arm[arm][metric] for arm in ARMS]
        effects[metric] = {"P2_without_LA": b - a, "P2_with_LA": d - c,
                           "LA_without_P2": c - a, "LA_with_P2": d - b,
                           "joint_vs_baseline": d - a, "interaction": d - b - c + a}
    (output / "effects.json").write_text(json.dumps(effects, indent=2, allow_nan=False) + "\n")
    lines = ["# 5090 P2＋P5-LA 四组配对结果", "", "AP为百分数；差值为百分点。延迟为未融合FP32、batch1、640纯GPU forward。",
             "P5包含在Neck/Encoder内；分段带标记，整网独立无标记采样。", "",
             "| 组别 | AP | APs | 整网 ms | P5 ms | Neck/Encoder ms |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['arm']} | {row['AP_percent']:.3f} | {row['APs_percent']:.3f} | "
                     f"{row['whole_ms']:.4f} | {row['p5_ms']:.4f} | {row['neck_encoder_ms']:.4f} |")
    lines += ["", "[可读数据](comparison.csv) · [配对差值及交互项](effects.json)", "",
              "单seed、固定预算结果；150轮不代表收敛，PyTorch速度不外推为TensorRT/Orin速度。",
              "归档时还需按项目AGENTS.md附完整必要曲线、初始化、环境、数据/源码身份及紧凑原始计时；权重和逐图预测留服务器。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluations", type=Path, nargs=4, required=True, metavar=("C0", "C2", "LA", "C2_LA"))
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report({arm: json.loads(path.read_text()) for arm, path in zip(ARMS, args.evaluations)},
           json.loads(args.benchmark.read_text()), args.output)
