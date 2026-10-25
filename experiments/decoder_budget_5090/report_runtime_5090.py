"""Render and validate the runtime archive with Python's standard library only."""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "docs/experiment_reports/2026-10-25_5090_runtime_profile"
LABELS = {"eager_fp32": "Eager FP32", "eager_fp16": "Eager FP16", "eager_bf16": "Eager BF16",
          "fused_fp32": "融合 FP32", "fused_fp16": "融合 FP16", "fused_bf16": "融合 BF16",
          "graph_fp32": "Graph FP32（未融合）", "graph_fused_fp32": "Graph + 融合 FP32",
          "graph_fused_fp16": "Graph + 融合 FP16"}


def load(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def csv_write(path, rows):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def validate(archive):
    raw = archive / "raw"
    identity = load(raw/"identity.json")
    require(set(identity["runtime"]) == {"c0", "c2"}, "Both model identities are required")
    for variant, provenance in identity["runtime"].items():
        model = load(raw/f"runtime-{variant}.json")
        require(model["hooks_remaining"] == 0, "Profiler hooks leaked")
        require(provenance["protocol"]["queries"] == 300 and provenance["protocol"]["points_per_level"] == [4,4,4], "Budget changed")
        expected = provenance["protocol"]["rounds"]*provenance["protocol"]["repeats_block"]
        for name, condition in model["conditions"].items():
            if condition["status"] != "completed":
                continue
            require(len(condition["blocks"]) == provenance["protocol"]["rounds"], "Timing rounds incomplete")
            require({b["round"] for b in condition["blocks"]} == set(range(1,provenance["protocol"]["rounds"]+1)), "Duplicate round")
            for metric in ("cuda", "wall"):
                samples = [v for b in condition["blocks"] for v in b[metric+"_ms"]]
                require(len(samples) == expected and all(math.isfinite(v) and v > 0 for v in samples), "Invalid raw timings")
                saved = condition[metric+"_stats_ms"]
                require(saved["n"] == expected and math.isclose(statistics.fmean(samples), saved["mean"], abs_tol=1e-9)
                        and math.isclose(statistics.median(samples), saved["median"], abs_tol=1e-9), "Timing summary differs from samples")
            if condition.get("cuda_graph"):
                require(condition["graph_vs_own_eager"]["max_abs"] == 0 and condition["changed_input_check"], "Graph equivalence failed")
    for path in raw.glob("profile-*.json"):
        p = load(path)
        iterations = p["iterations"]
        require(len(iterations) == 3, "Profiler samples incomplete")
        total_kernels = sum(i["kernel_count"] for i in iterations)
        total_us = sum(i["kernel_sum_us"] for i in iterations)
        for table, rows in p["tables"].items():
            require(sum(r["calls"] for r in rows) == total_kernels, f"Kernel count not conserved in {path.name}:{table}")
            require(math.isclose(sum(r["cuda_us"] for r in rows), total_us, abs_tol=.001), "Kernel time not conserved")
            require(all(math.isclose(r["cuda_us_per_forward"],r["cuda_us"]/3,abs_tol=1e-8) for r in rows), "Invalid profile average")
        for i in iterations:
            require(0 < i["gpu_busy_union_us"] <= i["gpu_span_us"] and
                    math.isclose(i["gpu_busy_union_us"]+i["gpu_idle_in_span_us"],i["gpu_span_us"],abs_tol=.001), "Invalid GPU timeline accounting")
    for path in raw.glob("deployment-*.json"):
        model = load(path)
        for c in model["conditions"].values():
            if c["status"] == "completed":
                a = [v for b in c["blocks"] for v in b["cuda_ms"]]
                require(len(a) == 250 and all(v>0 and math.isfinite(v) for v in a), "Deployment samples incomplete")
                require(math.isclose(statistics.fmean(a),c["cuda_stats_ms"]["mean"],abs_tol=1e-9), "Deployment summary differs")
                require(c["output_check"]["all_finite"], "Invalid deployment output")
                if "backend" in c and "providers" in c["backend"]:
                    require(c["backend"]["providers"][0] == "CUDAExecutionProvider", "Silent CPU fallback")
    return identity


def timeline(archive):
    rows = list(csv.DictReader((archive/"raw/timeline-c2.csv").open()))
    names = ["eager_fp32", "graph_fp32", "graph_fused_fp16"]
    colors = {"convolution":"#1565c0", "batch_norm":"#8e24aa", "matmul_linear_attention":"#00897b",
              "grid_sample":"#f57c00", "activation":"#e91e63", "other":"#78909c"}
    width, height, left, scale = 1200, 390, 215, .103
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font:12px sans-serif;fill:#263238}</style>',
             '<text x="15" y="22">C2: first profiled forward; profiler overhead included; common time scale</text>']
    for index,name in enumerate(names):
        selected = [r for r in rows if r["condition"] == name and r["mode"] == "plain"]
        y = 65+index*90
        parts.append(f'<text x="12" y="{y+12}">{html.escape(name)}</text>')
        parts.append(f'<text x="150" y="{y+7}">CPU launch</text><text x="165" y="{y+32}">CUDA</text>')
        launches = set()
        for r in selected:
            a,b = float(r["launch_start_us"]),float(r["launch_us"])
            if (a,b) not in launches:
                parts.append(f'<rect x="{left+max(0,a)*scale:.3f}" y="{y-4}" width="{max(.4,b*scale):.3f}" height="12" fill="#455a64"/>')
                launches.add((a,b))
            start,duration = float(r["gpu_start_us"]),float(r["gpu_us"])
            color = colors.get(r["group"],"#90a4ae")
            title = html.escape(f'{r["layer"]} {r["op"]} {duration:.3f} us')
            parts.append(f'<rect x="{left+start*scale:.3f}" y="{y+18}" width="{max(.18,duration*scale):.3f}" height="17" fill="{color}"><title>{title}</title></rect>')
    for ms in range(10):
        x = left+ms*1000*scale
        parts.append(f'<line x1="{x}" x2="{x}" y1="43" y2="294" stroke="#ddd" stroke-dasharray="2 4"/><text x="{x}" y="316">{ms} ms</text>')
    parts.append('<text x="15" y="349">White CUDA gaps are idle intervals. Graph kernels share one CPU graph launch; their operator labels are unavailable.</text>')
    parts.append('<text x="15" y="372">Blue: convolution; teal: matrix/attention; purple: BN; orange: grid_sample; grey: other or graph kernels.</text></svg>')
    (archive/"figures").mkdir(exist_ok=True)
    (archive/"figures/timeline.svg").write_text("\n".join(parts)+"\n")


def render(archive):
    identity = validate(archive)
    raw = archive/"raw"
    models = {v:load(raw/f"runtime-{v}.json") for v in ("c0","c2")}
    profile = lambda v,n,mode="plain": load(raw/models[v]["conditions"][n]["profile_"+mode])
    baseline = models["c2"]["conditions"]["eager_fp32"]["cuda_stats_ms"]["mean"]
    c2 = models["c2"]["conditions"]
    plain = profile("c2","eager_fp32")
    groups = {r["name"]:r for r in plain["tables"]["group"]}
    busy = statistics.fmean(i["gpu_busy_union_us"] for i in plain["iterations"])/1000
    graph = c2["graph_fp32"]["cuda_stats_ms"]["mean"]
    best = c2["graph_fused_fp16"]["cuda_stats_ms"]["mean"]
    deployment = load(raw/"deployment-c2.json")
    deploy_times = {n:c["cuda_stats_ms"]["mean"] for n,c in deployment["conditions"].items() if c["status"] == "completed"}
    lines = ["# RTX 5090：7.4 ms 的算子与运行时诊断", "",
             "2026-10-25，Asia/Shanghai；分支 `5090-detrscope-query-budget`。第一优先级推理诊断；未训练、未改 Query 或 Sampling Points。", "",
             f"C2 独立进程复测原生未融合 FP32 为 **{baseline:.3f} ms**。CUDA Graph 未融合 FP32 为 **{graph:.3f} ms**（延迟下降 {100*(1-graph/baseline):.1f}%）；融合 FP16 + Graph 为 **{best:.3f} ms**（下降 {100*(1-best/baseline):.1f}%）。这证明运行时与精度路径值得优先处理，同时 FP32 卷积仍是真实的大头。低精度与部署输出仅做 smoke 核验，不能承诺 AP 保持。", "",
             f"部署独立同轮对照：融合Eager FP32 **{deploy_times['eager_fused_fp32']:.3f} ms**，TensorRT FP32 **{deploy_times['trt_fp32']:.3f} ms**，TensorRT FP16 **{deploy_times['trt_fp16']:.3f} ms**。相同数学结构已能获得很大收益，建议先固定部署执行路径，再做Sampling Points的纯速度筛选。", "",
             "## 测量口径与原始 7.4 ms 的关系", "",
             "使用原 C0/C2 150e seed0 的 best.pt、原先同一张真实验证图，batch1、RGB 640×640、300Q、3 Decoder 层、8头、3尺度、每尺度4采样点。GPU输入/输出常驻；不含图片读取、预处理、H2D/D2H、后处理或NMS。输入与权重SHA、运行时源码SHA及环境均在 [identity.json](raw/identity.json)。", "",
             "PyTorch 2.14.0+cu130，RTX 5090 D v2 24GB，4 CPU线程，seed=20261024，TF32关闭，cuDNN benchmark关闭，inference_mode。每条件先预热150次；5轮随机交错条件，每块10次预热+50次计时，共250次。CUDA Event和同步到结束事件的CPU wall时间同时保留；无profiler、无module hook。Graph仅测replay，捕获不计时。", "",
             "C0采用首个模型的独立初始状态；C2另开独立进程复核，避免跨模型顺序/常驻对象影响。探索双模型同进程的C2曾为7.798 ms，该值不进入最终表；独立C2五轮FP32均值稳定约7.34 ms。不能将不同进程的C0/C2差值解释为结构收益。", "",
             "## 无 profiler 的整网性能", "",
             "| 路径 | C0 mean ms | C2 mean ms | C2 p95 ms | C2 wall mean ms | C2延迟下降 |", "|---|---:|---:|---:|---:|---:|"]
    speed_rows = []
    for name,label in LABELS.items():
        a,b = models["c0"]["conditions"][name],c2[name]
        require(a["status"] == b["status"] == "completed", "Runtime matrix incomplete")
        v = b["cuda_stats_ms"]
        lines.append(f"| {label} | {a['cuda_stats_ms']['mean']:.3f} | {v['mean']:.3f} | {v['p95']:.3f} | {b['wall_stats_ms']['mean']:.3f} | {100*(1-v['mean']/baseline):.1f}% |")
        for variant,c in (("c0",a),("c2",b)):
            speed_rows.append({"variant":variant,"condition":name, **{k+"_ms":v for k,v in c["cuda_stats_ms"].items() if k != "n"},
                               "samples":c["cuda_stats_ms"]["n"],"wall_mean_ms":c["wall_stats_ms"]["mean"]})
    csv_write(archive/"runtime.csv",speed_rows)
    lines += ["", "融合包含 Conv-BN 折叠与 RepConv 重参数化；原生69个BN全部消除。框架原生 model.fuse() 留下3个Decoder输入投影BN，额外的Sequential Conv-BN折叠单列在原始记录 stock_fusion_probe 与 fusion 字段；两者不能混称为同一种融合。FP16/BF16采用整模型与输入转换，autocast=False；Graph低精度为FP16。", "",
              "原生Graph捕获失败：AIFI在CPU创建位置编码并上传，Head也在CPU创建batch索引。诊断适配器在CUDA默认device上下文内运行捕获，固定形状常量在GPU生成；未改变参数、结构或预算。GPU三角函数舍入使适配FP32与原生最大绝对差约6e-6，Graph与其自身Eager输出逐值相同；输入乘0.75后也逐值一致。Graph适配不是已集成的动态部署接口，应用新输入仍需填充固定buffer。", "",
              "## CUDA Timeline 与 CPU 提交", "", "![C2 profiler timeline](figures/timeline.svg)", "",
              "上图展示各路径首个profiler forward，CPU launch与GPU kernel使用同一横轴。可悬停SVG条块查看算子。完整Chrome trace只保留在服务器，可用Perfetto或chrome://tracing打开；路径、大小、SHA见每个profile JSON的trace字段。仓库保留 [逐kernel紧凑timeline](raw/timeline-c2.csv) 与 [CPU算子/模块时间](raw/cpu-c2.csv)。", "",
              "| C2 profiler路径 | kernels/forward | kernel忙碌 ms | GPU首末kernel跨度 ms | 跨度内空档 | CPU提交间隔median/p95 µs |", "|---|---:|---:|---:|---:|---:|"]
    for name,mode,label in (("eager_fp32","plain","Eager FP32"),("eager_fp32","modules","Eager + module hooks"),
                            ("fused_fp32","plain","融合 FP32"),("graph_fp32","plain","Graph FP32"),
                            ("graph_fused_fp16","plain","Graph融合FP16")):
        p = profile("c2",name,mode)
        items = p["iterations"]
        avg = lambda key: statistics.fmean(i[key] for i in items)
        gaps = [i["cpu_launch_gap_us"] for i in items if i["cpu_launch_gap_us"]]
        gap = f"{statistics.fmean(i['median'] for i in gaps):.2f}/{statistics.fmean(i['p95'] for i in gaps):.2f}" if gaps else "单次Graph launch"
        lines.append(f"| {label} | {avg('kernel_count'):.0f} | {avg('gpu_busy_union_us')/1000:.3f} | {avg('gpu_span_us')/1000:.3f} | {100*avg('gpu_idle_fraction'):.1f}% | {gap} |")
    lines += ["", "profiler同时采CPU/CUDA，每路径3次forward，关闭shape/stack/memory追踪。无标签采样负责整体kernel/空档统计；另一轮module hooks负责归属。CPU列是含子调用的inclusive时间，不能把父子项相加。CPU发射间隔可能含CPU计算、同步和驱动等待，并非纯Python耗时。CUDA表按runtime correlation将每个kernel归属一次，避免卷积父算子与cuDNN子算子重复计时。", "",
              f"**不能将profiler内空档比例直接代入原生{baseline:.3f} ms。** profiler及module hooks本身放大CPU开销；模块标签路径空档更多。{baseline:.3f}→{graph:.3f} ms的无profiler同精度控制才是提交开销的实际收益证据（约{baseline-graph:.3f} ms）。Graph还改变常量生成的位置，故这不是严格的纯Python耗时测量。", "",
              "Graph未融合FP32的kernel数没有减少，Graph只压缩提交空档；FP16 cuDNN可能拆出更多kernel，数量本身不代表慢。FP32卷积仍需真正执行，不能把全部7.4 ms归因Python。GPU空档按首末kernel区间做时间并集；不包含forward起止边界、memcpy或memset。", "",
              "## GPU 算子花在哪里", "",
              f"C2无module标签profiler平均GPU kernel忙碌 **{busy:.3f} ms**；其中卷积 **{groups['convolution']['cuda_us_per_forward']/1000:.3f} ms**，占kernel总时长约 **{100*groups['convolution']['cuda_us']/sum(r['cuda_us'] for r in groups.values()):.1f}%**。这里是GPU kernel时间，不能当成原生forward独立逐项相加到7.4 ms。", "",
              "| 算子组 | kernels/forward | CUDA kernel ms/forward |", "|---|---:|---:|"]
    for r in plain["tables"]["group"]:
        lines.append(f"| {r['name']} | {r['calls_per_forward']:.0f} | {r['cuda_us_per_forward']/1000:.4f} |")
    csv_write(archive/"operators-c2.csv",plain["tables"]["op"])
    labeled = profile("c2","eager_fp32","modules")
    layers = labeled["tables"]["layer"]
    csv_write(archive/"layers-c2.csv",layers)
    csv_write(archive/"layer-operators-c2.csv",labeled["tables"]["layer_group"])
    lines += ["", "| C2模块（独占kernel归属） | kernels/forward | CUDA kernel ms/forward |", "|---|---:|---:|"]
    for r in sorted(layers,key=lambda r:int(r["name"].split(":")[0].split(".")[1])):
        lines.append(f"| {r['name']} | {r['calls_per_forward']:.0f} | {r['cuda_us_per_forward']/1000:.4f} |")
    lines += ["", "Neck/Encoder的层5–23不等于单个RepC3。四个RepC3中P3层16最大；AIFI的Attention（model.6.ma）、FFN（fc1/fc2/GELU）及LayerNorm均已分开，详见下面表格。P2层14为160×160输入到80×80的3×3步长2卷积，不是将整个P3 RepC3成本都归给P2。C0与C2的P3融合通道也不同，禁止只凭两网差值给P2归因。", "",
              "| AIFI子模块 | CUDA kernel ms/forward |", "|---|---:|"]
    for r in labeled["tables"]["module"]:
        if r["name"].startswith("model.6"):
            lines.append(f"| {r['name']} | {r['cuda_us_per_forward']/1000:.4f} |")
    lines += ["", "Neck层5–23按算子组再分解（同一module标签采样；为kernel忙碌时间，不是旧2.609 ms的CUDA流跨度）：", "",
              "| Neck算子组 | CUDA kernel ms/forward |", "|---|---:|"]
    for group in sorted({r["group"] for r in labeled["tables"]["layer_group"]}):
        duration = sum(r["cuda_us_per_forward"] for r in labeled["tables"]["layer_group"]
                       if 5 <= int(r["layer"].split(":")[0].split(".")[1]) <= 23 and r["group"] == group)
        if duration:
            lines.append(f"| {group} | {duration/1000:.4f} |")
    lines += ["", "| RepC3层 | Conv ms | BN ms | 激活 ms | 其他 ms |", "|---|---:|---:|---:|---:|"]
    for index in (11,16,19,22):
        selected = {r["group"]:r["cuda_us_per_forward"]/1000 for r in labeled["tables"]["layer_group"] if r["layer"].startswith(f"model.{index}:")}
        lines.append(f"| {index} | {selected.get('convolution',0):.4f} | {selected.get('batch_norm',0):.4f} | {selected.get('activation',0):.4f} | {sum(v for k,v in selected.items() if k not in ('convolution','batch_norm','activation')):.4f} |")
    lines += ["", "[RepC3各层Conv/BN/激活交叉表](layer-operators-c2.csv) · [完整算子表](operators-c2.csv) · [完整模块表](layers-c2.csv)。复制、布局变换与小逐元素算子也保留，未隐藏在Attention的总和中。注意MultiheadAttention部分工作由函数式linear完成，归属ma而非其未直接调用的out_proj模块。", "",
              "本次没有采SM吞吐或DRAM硬件计数器，不能据此断言所有卷积是算力受限还是带宽受限。FP32关闭TF32、低精度大幅加速、Graph同精度仍有收益，支持先优化数值执行路径、融合与提交，再评估结构改动。", "",
              "## ONNX / TensorRT", "",
              "独立进程中的原生融合FP32作为同轮桥接基准；不得把部署进程和上面runtime进程的亚百分比差异当收益。固定shape ONNX opset17，最终框/分数[1,300,14]，torch.onnx legacy导出期间关闭MHA fastpath，之后恢复。ONNX I/O binding与TensorRT execute_async_v3使用与CUDA Event相同的stream及GPU常驻地址，导出/build不计时。TensorRT关闭TF32；FP16 flag允许内部FP16/FP32混合，不声称全算子FP16。", "",
              "| 模型 | 部署路径 | mean ms | p95 ms | wall mean ms | 相对同轮融合FP32延迟下降 |", "|---|---|---:|---:|---:|---:|"]
    for path in sorted(raw.glob("deployment-*.json")):
        variant = path.stem.removeprefix("deployment-")
        m = load(path)
        bridge = m["conditions"]["eager_fused_fp32"]["cuda_stats_ms"]["mean"]
        for name,c in m["conditions"].items():
            if c["status"] == "completed":
                s = c["cuda_stats_ms"]
                lines.append(f"| {variant.upper()} | {name} | {s['mean']:.3f} | {s['p95']:.3f} | {c['wall_stats_ms']['mean']:.3f} | {100*(1-s['mean']/bridge):.1f}% |")
            else:
                lines.append(f"| {variant.upper()} | {name} | failed | — | — | {c.get('error','')} |")
    lines += ["", "| C2部署profiler路径 | kernels/forward | GPU kernel忙碌 ms | GPU首末kernel跨度 ms |", "|---|---:|---:|---:|"]
    for name,c in deployment["conditions"].items():
        if c["status"] == "completed" and "profile_plain" in c:
            p = load(raw/c["profile_plain"])
            rows = p["iterations"]
            avg = lambda key: statistics.fmean(r[key] for r in rows)
            lines.append(f"| {name} | {avg('kernel_count'):.0f} | {avg('gpu_busy_union_us')/1000:.3f} | {avg('gpu_span_us')/1000:.3f} |")
    lines += ["", "TensorRT FP32不仅减少CPU发射，也缩短GPU kernel忙碌时间；kernel名称包含FP32 Winograd及NHWC卷积实现。这提示后端tactic、布局与融合也比直接改Query/采样预算更值得先验证，不能把全部部署收益归因Python。具体kernel名称与次数保留在deployment profile JSON中。", "",
              "最终部署使用非默认CUDA stream；早期默认stream的额外同步警告已消除。TensorRT FP16构建仍出现6个Constant超FP16范围并裁剪到±65504的警告，输出虽finite，仍需正式AP/数值验证。ONNX EP设备注册警告、旧包安装失败及恢复方式见 [后端记录](raw/backend_notes.json)，不隐去这些负面证据。"]
    lines += ["", "ONNX必须实际启用CUDAExecutionProvider；禁止静默用CPU结果冒充GPU。CPU EP可执行形状控制等小操作，实际CUDA kernel trace与后端provider配置均归档。权重、ONNX、engine、完整trace与inspector JSON只留服务器，报告JSON记录路径/大小/SHA。", "",
              "## 数值检查与解释边界", "",
              "没有重新训练、没有重跑完整COCO评估。原有FP32正式AP记录仍属于原协议；本次FP16/BF16/TensorRT结果只用于性能。每条路径核验输出shape及finite；Graph额外核验与自身Eager逐值相同及更换输入。融合/低精度/部署对原生FP32的max/mean绝对差保存在各condition中。低精度会改变Top-K排序，逐query槽位差异可能很大，不能把该max差当定位退化，也不能据输出finite声称保AP。", "",
              "固定单图和静态形状不代表548图端到端延迟分布或动态batch吞吐；显存peak含常驻模型/Graph池，不能当单模型部署需求。此处选择Graph满足运行时对照，未额外测torch.compile。channels_last、TF32、cuDNN autotune与生产输入管理尚未调优，不声称1.9 ms已经是全局最优。", "",
              "## 对第二优先级的结论", "",
              f"当前9个grid_sample kernel合计仅 **{groups['grid_sample']['cuda_us_per_forward']/1000:.4f} ms**；它不包含MSDeformAttn的value/output投影、offset、weight与reshape等全部成本，不能把它当整个MSDeformAttn，也不能用它直接推算真实减点收益。现阶段先固定融合/低精度/Graph或TensorRT路径，再做用户提出的随机兼容模块速度筛选才有解释力。**本次未开展Sampling Points性能实验或训练。**", "",
              "## 复现与云端校验", "",
              "复测依赖identity.json列出的服务器checkpoint及同一验证图/标注；离线报告重建、哈希与相对链接检查只需仓库，不访问这些资产。复现命令与资产边界见 [入口README](README.md)。报告由 [report_runtime_5090.py](../../../experiments/decoder_budget_5090/report_runtime_5090.py) 基于raw文本生成。", ""]
    (archive/"REPORT.md").write_text("\n".join(lines))
    timeline(archive)


def manifest(archive):
    sources = [ROOT/"experiments/decoder_budget_5090"/name for name in (
        "profile_runtime_5090.py","profile_deployment_5090.py","archive_runtime_5090.py","report_runtime_5090.py",
        "tests/test_runtime_profile_5090.py")]
    files = sorted([p for p in archive.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"]+sources)
    (archive/"MANIFEST.sha256").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}\n" for p in files))


def verify(archive):
    validate(archive)
    for line in (archive/"MANIFEST.sha256").read_text().splitlines():
        digest,relative = line.split("  ",1)
        path = (ROOT/relative).resolve()
        require(path.is_relative_to(ROOT) and hashlib.sha256(path.read_bytes()).hexdigest() == digest, "SHA mismatch: "+relative)
    for p in archive.glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)",p.read_text()):
            if not target.startswith(("http:","https:")):
                target = target.split("#",1)[0]
                require((p.parent/target).exists(), "Broken link: "+target)
    print("Verified raw timing samples, kernel accounting, Graph equivalence, cloud manifest and report links")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",default=ARCHIVE,type=Path)
    p.add_argument("--verify",action="store_true")
    args = p.parse_args()
    if args.verify:
        verify(args.archive)
    else:
        render(args.archive)
        manifest(args.archive)


if __name__ == "__main__":
    main()
