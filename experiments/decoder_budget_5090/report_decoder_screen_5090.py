"""Offline decoder screen report/validation; Python standard library only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT/"docs/experiment_reports/2026-10-25_5090_decoder_screen"
BACKENDS = ("eager_fp32","graph_fp32","graph_fp16","trt_fp32","trt_fp16")
SAMPLING = ("q300-p4","q300-p2","q200-p4","q200-p2","q300-p1")


def read(path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def close(a,b):
    return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9)


def csv_write(path,rows):
    with path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load(archive):
    raw = archive/"raw"
    index = read(raw/"index.json")
    data = {"performance":{},"identity":{},"evaluation":{}}
    for filename in index["performance"]:
        saved = read(raw/filename)
        identity = read(raw/saved["identity"])
        for group in saved["groups"].values():
            key = (identity["kind"],identity["variant"],group["workload"])
            data["performance"].setdefault(key,{}).update({group["case"]:group["backends"]})
            data["identity"][(identity["kind"],identity["variant"])] = identity
    for filename in index["evaluation"]:
        saved = read(raw/filename)
        data["evaluation"][saved["variant"]] = saved
    return data


def validate(archive):
    data = load(archive)
    for variant in ("c0","c2"):
        for kind,workloads,cases in (("sampling",("model","attention"),SAMPLING),("depth",("model",),("d1","d2","d3"))):
            identity = data["identity"][(kind,variant)]
            p = identity["protocol"]
            require(p["rounds"]==10 and p["repeats_block"]==50,"Incomplete timing protocol")
            require(identity["hooks_remaining"]==0,"Profiling hooks leaked")
            for case,condition in identity["conditions"].items():
                if kind=="depth":
                    require(condition["queries"]==300 and condition["points_per_level"]==[4,4,4]
                            and condition["depth"]==int(case[1:]),"Depth comparison changed another budget")
                require(len(condition["executed"])==condition["depth"],"Wrong native execution depth")
                require(all(r["queries"]==condition["queries"] and r["points"]==condition["points_per_level"][0]
                            for r in condition["executed"]),"Budget was not executed")
                if kind=="sampling":
                    points = condition["points_per_level"][0]
                    require(len(condition["attention_inventory"])==3,"Missing replacement attention")
                    require(all(r["offset_outputs"]==8*3*points*2 and r["weight_outputs"]==8*3*points
                                for r in condition["attention_inventory"]),"Sampling projection dimensions unchanged")
            for workload in workloads:
                groups = data["performance"][(kind,variant,workload)]
                require(set(groups)==set(cases),"Performance cases incomplete")
                for backends in groups.values():
                    require(set(backends)==set(BACKENDS),"Execution paths incomplete")
                    for name,b in backends.items():
                        require(b["status"]=="completed","Failed backend cannot enter performance table")
                        require(len(b["blocks"])==10 and {v["round"] for v in b["blocks"]}==set(range(1,11)),"Timing rounds incomplete/duplicated")
                        samples = [v for block in b["blocks"] for v in block["cuda_ms"]]
                        require(all(len(block["cuda_ms"])==50 for block in b["blocks"]) and len(samples)==500
                                and all(v>0 and math.isfinite(v) for v in samples),"Invalid CUDA timing samples")
                        require(close(statistics.fmean(samples),b["cuda_stats_ms"]["mean"])
                                and close(statistics.median(samples),b["cuda_stats_ms"]["median"]),"Timing summary differs from samples")
                        require(b["cuda_stats_ms"]["n"]==500 and close(sorted(samples)[475],b["cuda_stats_ms"]["p95"]),"Invalid percentile/sample count")
                        wall = statistics.fmean(block["wall_stats_ms"]["mean"] for block in b["blocks"])
                        require(close(wall,b["wall_stats_ms"]["mean"]),"Pooled wall mean differs from blocks")
                        require(b["output_check"]["all_finite"],"Nonfinite output")
                        if name.startswith("graph"):
                            require(b["graph_exact_own_eager"] and b["changed_input_check"],"Graph gate failed")
                        if name.startswith("trt"):
                            require(b["engine"]["tf32"] is False,"Unexpected TF32")
                            require(len(b["engine"]["input_names"])==(3 if workload=="attention" else 1),"Constant-folded microbenchmark")
        evaluation = data["evaluation"][variant]
        require(evaluation["status"]=="completed" and evaluation["protocol"]["images"]==548
                and evaluation["protocol"]["full_validation"],"Subset cannot masquerade as full AP evaluation")
        require(evaluation["early_exit_equivalence"]["all_exact"] and evaluation["early_exit_equivalence"]["images"]==548
                and all(v==0 for v in evaluation["early_exit_equivalence"]["max_abs"].values()),"Native early-exit equivalence failed")
        require(evaluation["observation"]["all_identity_and_restore_passed"] and evaluation["model_state_unchanged"]
                and evaluation["hooks_remaining"]==0 and evaluation["eval_idx_restored"]==2,"Native observation restoration failed")
        require(evaluation.get("historical_d3_replay",{}).get("all_metrics_exact"),"D3 historical AP replay not exact")
        require(evaluation["diagnostics"]["images"]==548,"Trajectory diagnostics incomplete")
        require(evaluation["formal_protocol"]["max_dets"]==[1,10,100],"Wrong official AP maxDets")
        require(evaluation["checkpoint"]["sha256"]==data["identity"][("sampling",variant)]["checkpoint"]["sha256"]
                ==data["identity"][("depth",variant)]["checkpoint"]["sha256"],"Performance/AP checkpoints differ")
        with (archive/"raw"/(variant+"-formal_gt_transitions.csv")).open() as stream:
            for row in csv.DictReader(stream):
                require(sum(int(row[k]) for k in ("lost","gained","stable_tp","stable_fn"))==int(row["gt_denominator"]),"GT states do not partition denominator")
                if row["group"]=="all":
                    require(int(row["gt_denominator"])==evaluation["conditions"]["d3"]["gt_denominator"],"GT denominator changed")
    return data


def reduction(before,after):
    """Paired round/block bootstrap; never treat 500 within-block samples as independent."""
    a = {b["round"]:statistics.fmean(b["cuda_ms"]) for b in before["blocks"]}
    b = {b["round"]:statistics.fmean(b["cuda_ms"]) for b in after["blocks"]}
    require(a.keys()==b.keys(),"Unpaired timing rounds")
    ids = sorted(a)
    point = 100*(1-after["cuda_stats_ms"]["mean"]/before["cuda_stats_ms"]["mean"])
    rng = random.Random(20261025)
    boot = []
    for _ in range(4000):
        sample = rng.choices(ids,k=len(ids))
        boot.append(100*(1-sum(b[i] for i in sample)/sum(a[i] for i in sample)))
    boot.sort()
    return {"latency_reduction_percent":point,"ci95_low":boot[100],"ci95_high":boot[3899],"paired_rounds":len(ids)}


def verify(archive):
    validate(archive)
    for line in (archive/"MANIFEST.sha256").read_text().splitlines():
        digest,name = line.split("  ",1)
        path = (ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and hashlib.sha256(path.read_bytes()).hexdigest()==digest,"Manifest mismatch: "+name)
    for path in archive.glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)",path.read_text()):
            if not target.startswith(("http:","https:")):
                target = target.split("#",1)[0]
                require((path.parent/target).exists(),"Broken link: "+target)
    print("Verified CUDA samples, actual budgets, engine inputs, native early-exit/AP identity, GT accounting and cloud links")


def manifest(archive):
    sources = [ROOT/"experiments/decoder_budget_5090"/name for name in (
        "screen_decoder_perf_5090.py","screen_decoder_eval_5090.py","archive_decoder_screen_5090.py",
        "report_decoder_screen_5090.py","tests/test_decoder_screen_5090.py")]
    files = sorted([p for p in archive.rglob("*") if p.is_file() and p.name!="MANIFEST.sha256"]+sources)
    (archive/"MANIFEST.sha256").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p.relative_to(ROOT))+"\n" for p in files))


def render(archive):
    data = validate(archive)
    perf = data["performance"]
    evaluation = data["evaluation"]
    sampling_rows,depth_rows,comparisons = [],[],[]
    for (kind,variant,workload),cases in sorted(perf.items()):
        baseline = cases["q300-p4" if kind=="sampling" else "d3"]
        for case,backends in cases.items():
            for backend,condition in backends.items():
                s = condition["cuda_stats_ms"]
                r = reduction(baseline[backend],condition)
                row = {"variant":variant,"case":case,"workload":workload,"backend":backend,
                       "mean_ms":s["mean"],"median_ms":s["median"],"p95_ms":s["p95"],"forward_fps":1000/s["mean"],
                       "samples":s["n"],"wall_mean_ms":condition["wall_stats_ms"]["mean"],**r}
                (sampling_rows if kind=="sampling" else depth_rows).append(row)
        if kind=="sampling":
            for before,after,label in (("q300-p4","q300-p2","points_only_q300"),("q200-p4","q200-p2","points_only_q200"),
                                       ("q300-p4","q200-p4","queries_only"),("q300-p4","q200-p2","joint"),("q300-p4","q300-p1","extreme_points")):
                for backend in BACKENDS:
                    comparisons.append({"variant":variant,"workload":workload,"comparison":label,"backend":backend,
                                        "before":before,"after":after,**reduction(cases[before][backend],cases[after][backend])})
    csv_write(archive/"sampling_performance.csv",sampling_rows)
    csv_write(archive/"sampling_comparisons.csv",comparisons)
    csv_write(archive/"depth_performance.csv",depth_rows)
    ap_rows = []
    for variant,summary in evaluation.items():
        baseline = summary["conditions"]["d3"]["metrics"]
        for case,condition in summary["conditions"].items():
            ap_rows.append({"variant":variant,"depth":condition["depth"],**condition["metrics"],
                            **{"delta_"+k+"_pp":100*(v-baseline[k]) for k,v in condition["metrics"].items()}})
    csv_write(archive/"depth_ap.csv",ap_rows)
    c2 = perf[("sampling","c2","model")]
    half = reduction(c2["q300-p4"]["trt_fp16"],c2["q300-p2"]["trt_fp16"])
    d2 = evaluation["c2"]["conditions"]["d2"]["metrics"]
    d3 = evaluation["c2"]["conditions"]["d3"]["metrics"]
    timing = perf[("depth","c2","model")]
    depth_gain = reduction(timing["d3"]["trt_fp16"],timing["d2"]["trt_fp16"])
    verdict = ("单独减点未达到5%的整网延迟下降，当前不建议立即启动多组长周期训练。" if half["latency_reduction_percent"]<5 else
               "单独减点已达到5%的整网延迟下降，具备后续少量保AP实验的性能动机；本轮不据此判断精度。")
    lines = ["# 5090：Sampling Points 纯性能筛选与 Decoder 深度评估", "",
             "2026-10-25（Asia/Shanghai）；分支 `5090-detrscope-query-budget`。使用原C0/C2各150e、seed0的best.pt；本次未训练。", "",
             f"第二优先级：C2在TensorRT FP16执行路径上，300Q采样点由[4,4,4]降到[2,2,2]，整网延迟下降 **{half['latency_reduction_percent']:.2f}%**，配对轮次bootstrap 95%区间 **[{half['ci95_low']:.2f}%, {half['ci95_high']:.2f}%]**。{verdict}", "",
             f"第三优先级：C2原生FP32正式COCO AP，D2为 **{100*d2['AP']:.3f}%**，D3为 **{100*d3['AP']:.3f}%**；D2相对D3为 **{100*(d2['AP']-d3['AP']):+.3f}个百分点**。TensorRT FP16早退D2的整网延迟下降 **{depth_gain['latency_reduction_percent']:.2f}%**。精度与性能使用的数值路径分开记录，尚不能把FP32 AP直接当成FP16部署AP。", "",
             "## 输入、实现与统计口径", "",
             "性能为5090 D v2、batch1、RGB 640×640、静态shape、GPU输入/输出常驻的forward；不包含图片I/O、H2D/D2H、预处理、后处理或NMS。固定原先同一张真实验证图；MSDeformAttn使用原生D1实际query/reference/value张量，256维、8头、3尺度80×80/40×40/20×20，8400个value token，reference末维4。300/200Q只是query长度，value长度保持8400。", "",
             "每模型独立进程，无profiler/observer参与计时。各条件150次预热，再10轮随机交错顺序，每块10次预热与50次CUDA Event计时，合计500次；CPU同步wall统计同时保留。Graph使用GPU常量创建、Conv-BN/RepConv融合，FP16整模型/输入转换；TensorRT固定opset17，TF32关闭，FP16允许内部混合精度、FP32 I/O，统一非默认stream。引擎构建、捕获、导出均不计时。", "",
             "延迟下降=1−候选ms/基准ms，不是FPS增长百分比。95%区间按10个配对轮次做4000次确定性bootstrap，不把500个轮内计时视为500个独立实验。区间只反映本轮计时噪声，不覆盖重新构建TensorRT tactic、其他图片/采样位置或不同硬件的变化；每条件仅构建一份引擎。", "",
             "## 第二优先级：Sampling Points", "",
             "所有五组，包括[4,4,4]基准，均将全部3层Cross-Attention换成同规则随机兼容模块；保留其余训练权重。value/output投影逐字节相同；offset/attention矩阵采用N(0,0.01)，按head/level取相同点前缀，避免默认零矩阵让优化器只测常量；offset bias保留原生环形初始化。真实缩小offset/weight线性输出维度与grid宽度，不用mask，也不在完整计算后丢点。", "",
             "4D reference仍执行原生offset/n_points×reference_wh×0.5，改变点数也改变采样坐标。随机模块改变数值与后续Decoder输入，**不运行/不报告这一组的AP**。独立模块ONNX把query、reference、value全部保留为运行时输入，禁止把value投影预先常量折叠掉。", "",
             "256维/8头/3尺度时，offset投影输出维度192→96→48，attention weight输出96→48→24；单个MSDeformAttn参数205600→168592→150088，三个模块全部减半点数共减少111024个参数。8400-token value投影、output投影、Self-Attention、FFN及Backbone/Neck并不随点数减半；采样位置数减半不能推算整个模块或整网耗时减半。", "",
             "| Q | 每尺度点数 | C0整网 TRT FP16 ms | C2整网 TRT FP16 ms | C2独立Attention TRT FP16 ms | C2整网延迟下降（相对300Q/p4） |", "|---:|---|---:|---:|---:|---:|"]
    for case in SAMPLING:
        cfg = data["identity"][("sampling","c2")]["conditions"][case]
        a = perf[("sampling","c0","model")][case]["trt_fp16"]["cuda_stats_ms"]["mean"]
        b = c2[case]["trt_fp16"]["cuda_stats_ms"]["mean"]
        micro = perf[("sampling","c2","attention")][case]["trt_fp16"]["cuda_stats_ms"]["mean"]
        r = reduction(c2["q300-p4"]["trt_fp16"],c2[case]["trt_fp16"])
        lines.append(f"| {cfg['queries']} | {cfg['points_per_level']} | {a:.4f} | {b:.4f} | {micro:.4f} | {r['latency_reduction_percent']:+.2f}% |")
    lines += ["", "单变量与联合效应分别统计，避免把200Q本身的收益算给减点：", "",
              "| 模型 | 工作量 | 对比 | TensorRT FP16延迟下降 | 配对95%区间 |", "|---|---|---|---:|---:|"]
    for r in comparisons:
        if r["backend"]=="trt_fp16":
            lines.append(f"| {r['variant'].upper()} | {r['workload']} | {r['comparison']} | {r['latency_reduction_percent']:+.2f}% | [{r['ci95_low']:.2f}%, {r['ci95_high']:.2f}%] |")
    lines += ["", "| C2整网执行路径 | 300Q/p4 ms | 300Q/p2 ms | 单独减点延迟下降 | 200Q/p4 ms | 200Q/p2 ms | 300Q/p1 ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for backend in BACKENDS:
        times = [c2[case][backend]["cuda_stats_ms"]["mean"] for case in SAMPLING]
        r = reduction(c2["q300-p4"][backend],c2["q300-p2"][backend])
        lines.append(f"| {backend} | {times[0]:.4f} | {times[1]:.4f} | {r['latency_reduction_percent']:+.2f}% | {times[2]:.4f} | {times[3]:.4f} | {times[4]:.4f} |")
    lines += ["", "[完整整网/独立模块性能表](sampling_performance.csv)含两模型、全部执行路径、median/p95/FPS；[全部配对比较](sampling_comparisons.csv)含条件200Q下的独立减点效应。原始CUDA逐次数据在raw中，未以图或压缩包替代。显存peak包含同进程常驻的多条候选路径/引擎/Graph池，不应解释为单模型部署显存或减点显存收益。", "",
              "## 第三优先级：D1/D2/D3 真实权重", "",
              "每个checkpoint先以原生D3运行DETRScope，观测全部层并逐batch验证bare/observed/restored输出及RNG完全一致。然后在全部548张图上分别令原生eval_idx=0/1/2重执行，hook验证只执行前1/2/3层；每张图的完整框/分数必须与D3观测的对应中间层逐值相同。不是裁剪D3的最终预测，也不是用D3分类头读取D1特征。D1/D2使用各自已训练的辅助分类头。", "",
              "AP使用原生未融合FP32、batch2、conf>0.001、每query argmax类别、按分数排序max_det=300、不舍入、不NMS；正式pycocotools COCOeval bbox，maxDets=[1,10,100]、IoU=.50:.05:.95。原标注/类别映射/图像顺序冻结。D3的全部12项指标必须逐值重放上一轮同协议300Q结果。", "",
              "| 模型 | 输出层 | 正式AP % | AP50 % | AP75 % | APs % | APm % | APl % | 相对D3 AP百分点 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ap_rows:
        lines.append(f"| {r['variant'].upper()} | D{r['depth']} | {100*r['AP']:.3f} | {100*r['AP50']:.3f} | {100*r['AP75']:.3f} | {100*r['APs']:.3f} | {100*r['APm']:.3f} | {100*r['APl']:.3f} | {r['delta_AP_pp']:+.3f} |")
    lines += ["", "深度性能使用原始真实权重，固定300Q/[4,4,4]，每条路径真实提前停止；没有改采样模块。下表TensorRT FP16供速度筛选，FP32 AP与其并列不代表FP16已通过完整精度验证。", "",
              "| 模型 | 输出层 | Eager FP32 ms | 融合Graph FP32 ms | 融合Graph FP16 ms | TensorRT FP32 ms | TensorRT FP16 ms | TRT FP16相对D3延迟下降 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for variant in ("c0","c2"):
        groups = perf[("depth",variant,"model")]
        for case in ("d1","d2","d3"):
            times = [groups[case][b]["cuda_stats_ms"]["mean"] for b in BACKENDS]
            gain = reduction(groups["d3"]["trt_fp16"],groups[case]["trt_fp16"])
            lines.append(f"| {variant.upper()} | {case.upper()} | "+" | ".join(f"{t:.4f}" for t in times)+f" | {gain['latency_reduction_percent']:+.2f}% |")
    lines += ["", "[全部深度性能与配对区间](depth_performance.csv) · [全部正式AP/AR及百分点差值](depth_ap.csv)。PyTorch中未删除未执行层，驻留参数量不变；导出引擎可裁掉未执行计算，不能把早退说成原checkpoint参数已物理删除。", "",
              "## 同一GT的状态迁移", "",
              "正式TP/FN迁移由COCOeval的匹配GT ID计算，按同一标注ID配对，不用query slot跨层/跨模型代替目标身份。每个IoU阈值保留Lost/Gained/StableTP/StableFN，四类之和逐行核验GT分母，另按原图面积与类别分组。迁移诊断的面积组为small<1024、medium=[1024,9216)、large≥9216，属于互斥分组，不替代正式COCO面积端点定义。这是给定IoU、maxDets=100的匹配状态，不能把它当AP差值。", "",
              "| 模型 | 迁移 | IoU | GT分母 | Lost | Gained | StableTP | StableFN |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for variant in ("c0","c2"):
        with (archive/"raw"/(variant+"-formal_gt_transitions.csv")).open() as stream:
            for r in csv.DictReader(stream):
                if r["group"]=="all" and float(r["iou"]) in (.5,.75):
                    lines.append(f"| {variant.upper()} | {r['transition']} | {r['iou']} | {r['gt_denominator']} | {r['lost']} | {r['gained']} | {r['stable_tp']} | {r['stable_fn']} |")
    lines += ["", "同时对全部548图复用DETRScope的Hungarian责任分配、固定query-GT轨迹与逐层重新分配GT轨迹，IoU/GT类别分数变化阈值均0.02。它不设正式TP的IoU/置信门槛，query switch属于诊断，和上面的COCO正式状态分开解释。诊断class_id为模型0基索引，正式迁移category为原COCO类别ID；映射保存在depth-eval协议中。分母、排除项、temporary/persistent degradation及尺度/类别统计全部保留在 [C0诊断CSV](raw/c0-diagnostic_transitions.csv)、[C2诊断CSV](raw/c2-diagnostic_transitions.csv)；[C0正式GT迁移](raw/c0-formal_gt_transitions.csv)、[C2正式GT迁移](raw/c2-formal_gt_transitions.csv)包含全部10个IoU阈值。", "",
              "## 结论边界与下一步", "",
              verdict,
              "D3对精度的贡献用上面的正式AP/APs差值和GT Lost/Gained共同判断；D2早退速度收益用同轮实际重执行判断。若后续选择D2，应另做少量真实微调/蒸馏与目标部署精度验收，不把当前早退直接等同于重训后的2层模型。单seed与单卡固定图计时无法证明其他设备、图片分布或多seed收益。", "",
              "本轮没有训练，也没有宣称随机缩点模块或低精度部署保AP。TensorRT FP16的常量范围裁剪、导出模式修复与失败批次见 [执行记录](raw/execution_notes.json)。所有原始权重/预测/逐GT列表/逐query大tensor、ONNX/engine与完整日志仍在服务器，原路径/大小/SHA见身份记录；云端只需当前raw文本即可重建、校验报告。复现命令和环境见 [入口README](README.md)。", ""]
    diagnostics = ["| 模型 | 轨迹口径 | 迁移 | 有效分母 | IoU下降% | IoU提升% | GT分数下降% | GT分数提升% | Query切换% |",
                   "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    c2_switch = 0.
    for variant in ("c0","c2"):
        with (archive/"raw"/(variant+"-diagnostic_transitions.csv")).open() as stream:
            rows = [r for r in csv.DictReader(stream) if r["class_id"]=="all" and r["object_size"]=="all"]
        for family in ("fixed_query_gt","gt_object"):
            for transition in ("D1->D2","D2->D3"):
                selected = {r["event"]:r for r in rows if r["family"]==family and r["transition"]==transition}
                n = selected["localization_drop"]["denominator"]
                rates = [100*float(selected[name]["rate"]) for name in
                         ("localization_drop","localization_gain","classification_drop","classification_gain")]
                switch = 100*float(selected["query_switch"]["rate"]) if "query_switch" in selected else None
                if variant=="c2" and family=="gt_object" and transition=="D2->D3":
                    c2_switch = switch
                switch_text = f"{switch:.2f}" if switch is not None else "—"
                diagnostics.append(f"| {variant.upper()} | {family} | {transition} | {n} | "+
                                   " | ".join(f"{r:.2f}" for r in rates)+f" | {switch_text} |")
    diagnostics += ["", f"C2从D2到D3的小目标AP增加 **{100*(d3['APs']-d2['APs']):.3f}个百分点**；GT责任轨迹有 **{c2_switch:.2f}%** 的query切换。固定query-GT对与重新分配GT轨迹应结合正式AP、Lost/Gained解读，不能用单一轨迹的下降数量替代最终检测收益。", ""]
    position = lines.index("## 结论边界与下一步")
    lines[position:position] = diagnostics
    (archive/"REPORT.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",type=Path,default=ARCHIVE)
    p.add_argument("--verify",action="store_true")
    args = p.parse_args()
    if args.verify:
        verify(args.archive)
    else:
        render(args.archive)
        manifest(args.archive)


if __name__ == "__main__":
    main()
