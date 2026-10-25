"""Small live/offline report for the four fresh training runs; no torch import."""
import argparse
import csv
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import tempfile
import os


def read(path):
    return json.loads(path.read_text())


def atomic(path,text):
    with tempfile.NamedTemporaryFile(mode="w",dir=path.parent,delete=False) as f:
        f.write(text);name=f.name
    os.replace(name,path)


def collect(root):
    queue=read(root/"status.json") if (root/"status.json").exists() else {}
    records=[]
    for path in sorted(root.glob("5090-*/experiment.json")):
        state=read(path)
        curve=[]
        if (path.parent/"results.csv").is_file():
            with (path.parent/"results.csv").open() as f:
                curve=list(csv.DictReader(f))
        evaluation=path.parent.parent/(path.parent.name+"-eval")/"summary.json"
        result=read(evaluation) if evaluation.is_file() else None
        records.append({"state":state,"curve":curve,"evaluation":result})
    return {"schema":"5090.fresh-budget-report.v1","generated_utc":datetime.now(timezone.utc).isoformat(),
            "root":str(root),"queue":queue,"records":records}


def render(data,output):
    lines=["# 5090 四组从零训练", "",f"快照时间（UTC）：`{data['generated_utc']}`；队列状态：`{data['queue'].get('status','unknown')}`。",
           "", "150e、seed0、batch8、640、AdamW、FP32；无pretrained、无旧权重微调、无旧checkpoint续训。", "",
           "此文件在每轮和阶段切换时本地自动刷新；Git云端只显示最近推送的快照，不是实时服务器状态。工程短跑指标不进入正式模型比较。", "",
           "| 实验 | 角色 | 训练状态 | 已完成轮数 | 内部mAP50–95 % | 正式AP % | APs % | TRT FP16 ms |", "|---|---|---|---:|---:|---:|---:|---:|"]
    rows=[]
    for item in data["records"]:
        s=item["state"];e=item["evaluation"] or {}
        metric=e.get("metrics",{}) if e.get("scope")=="full_val" and e.get("status")=="completed" else {}
        timing=e.get("benchmark",{}).get("backends",{}).get("trt_fp16",{}).get("cuda_stats_ms",{})
        row={"name":s["experiment_name"],"variant":s["variant"],"role":s["role"],"status":s["status"],
             "epochs":len(item["curve"]),"requested":s["epochs_requested"],"internal_map":s.get("best_internal_map"),
             "AP":metric.get("AP"),"APs":metric.get("APs"),"trt_fp16_ms":timing.get("mean"),
             "checkpoint_sha256":s.get("checkpoints",{}).get("best",{}).get("sha256")}
        rows.append(row)
        percent=lambda v:"—" if v is None else f"{100*v:.3f}"
        ms="—" if row["trt_fp16_ms"] is None else f"{row['trt_fp16_ms']:.4f}"
        lines.append(f"| {row['variant']} | {row['role']} | {row['status']} | {row['epochs']}/{row['requested']} | {percent(row['internal_map'])} | {percent(row['AP'])} | {percent(row['APs'])} | {ms} |")
    lines += ["", "正式AP与训练内部指标分开；正式AP为原生未融合FP32 COCOeval，TensorRT FP16只测性能。完整必要训练曲线保存在同目录experiments.json，checkpoint/原始日志/逐图预测仅服务器保存。", ""]
    atomic(output/"experiments.md","\n".join(lines))
    atomic(output/"experiments.json",json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    import io
    buf=io.StringIO()
    fields=list(rows[0]) if rows else ["name","variant","role","status","epochs","requested","internal_map","AP","APs","trt_fp16_ms","checkpoint_sha256"]
    writer=csv.DictWriter(buf,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    atomic(output/"experiments.csv",buf.getvalue())
    return data


def generate(root,output,data=None):
    root,output=Path(root).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    with (output/".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return render(collect(root) if data is None else data,output)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--from-json",type=Path)
    a=p.parse_args()
    if not a.root and not a.from_json:p.error("Provide --root or --from-json")
    generate(a.root or Path('.'),a.output,read(a.from_json) if a.from_json else None)
