"""Archive compact final evidence; predictions/checkpoints remain on the server."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def archive(root, output):
    root, output = root.resolve(), output.resolve()
    raw = output / "results"
    raw.mkdir(parents=True, exist_ok=False)
    run = root / "training/5090-c2-o2mc-150e-s0"
    state = json.loads((run / "experiment.json").read_text())
    if state["status"] != "completed" or state["completed_epochs"] != 150:
        raise ValueError("Require a complete 150-epoch receipt")
    items = {run / "experiment.json": "training.json", run / "results.csv": "training_curve.csv",
             run / "health.jsonl": "health.jsonl", run / "aux_diagnostics.jsonl": "aux_diagnostics.jsonl",
             run / "args.yaml": "effective_args.yaml", root / "plan.json": "plan.json",
             root / "asset_audit.json": "asset_audit.json", root / "comparison.json": "comparison.json",
             root / "evaluation/summary.json": "evaluation.json",
             root / "baseline-evaluation/summary.json": "baseline_evaluation.json"}
    for source, target in items.items():
        shutil.copy2(source, raw / target)
    render(output)


def render(output):
    """Regenerate the report using only the compact files shipped in Git."""
    output = output.resolve()
    raw = output / "results"
    state = json.loads((raw / "training.json").read_text())
    root = Path(state["checkpoints"]["best"]["path"]).parents[3]
    comparison = json.loads((raw / "comparison.json").read_text())
    rows = list(csv.DictReader((raw / "training_curve.csv").open()))
    lines = ["# C2 + O2M-C：150轮单 seed 结果", "",
             f"分支 `{state['source_branch']}`；基点 `{state['source_commit']}`，实际源码身份见 [training.json](results/training.json)。",
             "完整6471图训练、548图标准COCO评估。精度训练FP32，checkpoint原生保存为FP16 EMA后重载FP32；不是VisDrone官方挑战分数。", "",
             "| 指标 | C2基线 | C2+O2M-C | 差值pp |", "|---|---:|---:|---:|"]
    for key in ("AP", "AP50", "AP75", "APs", "APm", "APl"):
        lines.append(f"| {key} | {100 * comparison['baseline'][key]:.6f} | {100 * comparison['candidate'][key]:.6f} | {comparison['delta_pp'][key]:+.6f} |")
    lines += ["", f"训练曲线覆盖 {len(rows)} 轮；最佳内部mAP50–95={state['best_internal_map']:.6f}。内部验证用于选择best，表中为best完整正式COCO AP。",
              f"收敛检查：`{json.dumps(state['convergence'], ensure_ascii=False)}`；完成150轮不自动表示收敛。", "",
              "[完整训练曲线](results/training_curve.csv) · [每轮资源和健康](results/health.jsonl) · [分配与梯度诊断](results/aux_diagnostics.jsonl) · [分类别和尺度指标](results/evaluation.json)。", "",
              "该比较只回答seed0、150轮预算下辅助监督是否改善C2。不能推断多seed稳定性，未做C0-Aux，不能宣称C2与辅助监督协同。",
              "训练诊断面积是640输入坐标下的面积；COCO APs按原图标注面积，二者分母不能混用。每轮第三个batch的梯度只是抽样。", "",
              "权重、完整预测、源码快照与完整日志仅保存在服务器：", f"`{root}`。best/last/resume路径、大小及SHA见training.json；部署原生权重为服务器 `evaluation/native-c2.pt`。", "",
              "设计与复现见 [PLAN.md](PLAN.md)。"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    readme = output / "README.md"
    content = readme.read_text()
    begin, end = "<!-- RUN_STATUS_BEGIN -->", "<!-- RUN_STATUS_END -->"
    if begin in content and end in content:
        prefix, rest = content.split(begin, 1)
        _, suffix = rest.split(end, 1)
        readme.write_text(prefix + begin + "\n150轮训练和完整正式评估已完成，见 [最终结果](RESULTS.md)。\n" + end + suffix)
    paths = sorted(p for p in output.rglob("*") if p.is_file()
                   and p.name != "MANIFEST.sha256" and p.suffix != ".pyc")
    (output / "MANIFEST.sha256").write_text("".join(
        hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.relative_to(output).as_posix() + "\n" for p in paths))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--render-only", action="store_true", help="Use only archived results; no server data/checkpoints")
    args = p.parse_args()
    if args.render_only:
        render(args.output)
    else:
        if args.root is None:
            p.error("Archive requires --root")
        archive(args.root, args.output)
