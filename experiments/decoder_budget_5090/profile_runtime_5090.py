"""CPU/CUDA operator diagnosis and runtime controls; never trains or changes query/point budgets.

Run as a module from the repository root. Full Chrome traces stay in ignored runs/.
Compact timing samples and non-overlapping kernel attribution are archived separately.
"""
import argparse
from contextlib import contextmanager
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time
from zoneinfo import ZoneInfo

import torch

from experiments.decoder_budget_5090.run import configure, environment, source_identity

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "runs/decoder_budget_5090/5090-c0c2-baselines-20261024-s0-r1"
DEFAULT_DATA = ROOT.parent / "SO_DETR/datasets/VisDrone2019-DET-YOLO"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def stats(a):
    a = sorted(a)
    return {"mean": statistics.fmean(a), "median": statistics.median(a),
            "p95": a[min(len(a)-1, int(.95*len(a)))], "min": min(a), "max": max(a), "n": len(a)}


def clocks():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=clocks.current.sm,clocks.current.memory,pstate,temperature.gpu,utilization.gpu",
                                    "--format=csv,noheader"], text=True).strip()


def load_input(data):
    import cv2
    import numpy as np
    from ultralytics.data.augment import LetterBox
    ann = data / "annotations/VisDrone2019-DET_val_coco.json"
    image = sorted(json.loads(ann.read_text())["images"], key=lambda r: r["id"])[0]
    path = data / "images/val" / Path(image["file_name"]).name
    arr = LetterBox((640, 640), auto=False, scale_fill=True)(image=cv2.imread(str(path)))
    arr = np.ascontiguousarray(arr[..., ::-1].transpose(2, 0, 1)[None])
    x = torch.from_numpy(arr).cuda().float() / 255
    return x, {"annotation_path": str(ann), "annotation_sha256": sha(ann), "image": image,
               "image_path": str(path), "image_sha256": sha(path),
               "input_sha256": hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
               "preprocess": "OpenCV BGR -> scale_fill 640 square -> RGB BCHW contiguous FP32 /255; before timing"}


def prediction(y):
    return y if isinstance(y, torch.Tensor) else y[0]


def comparison(ref, actual):
    a, b = prediction(ref).float(), prediction(actual).float()
    if a.shape != b.shape or not torch.isfinite(b).all():
        raise ValueError("Output shape/finite check failed")
    d = (a-b).abs()
    return {"shape": list(b.shape), "all_finite": True, "max_abs": d.max().item(),
            "mean_abs": d.mean().item(), "box_max_abs": d[..., :4].max().item(),
            "score_max_abs": d[..., 4:].max().item(),
            "note": "Query slots may reorder after rounding; differences are a smoke diagnostic, not AP validation"}


def fuse(model, full=False):
    """Framework fusion includes RepConv reparameterization; full additionally folds head Sequential Conv-BN."""
    before = sum(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    model.fuse(verbose=False)
    if full:
        for module in list(model.modules()):
            if isinstance(module, torch.nn.Sequential):
                items = list(module._modules.items())
                for (name, a), (bn_name, b) in zip(items, items[1:]):
                    if isinstance(a, torch.nn.Conv2d) and isinstance(b, torch.nn.BatchNorm2d):
                        module._modules[name] = torch.nn.utils.fuse_conv_bn_eval(a, b)
                        module._modules[bn_name] = torch.nn.Identity()
    return {"bn_before": before, "bn_after": sum(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()),
            "repconv_reparameterized": True, "sequential_conv_bn_folded": full}


def capture(model, x):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            model(x)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = model(x)
    def replay():
        graph.replay()
        return output
    return replay, graph


def measure(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record(); end.record(); end.synchronize()
    cuda, wall = [], []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        t = time.perf_counter_ns()
        start.record()
        fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter_ns()-t)/1e6)
        cuda.append(start.elapsed_time(end))
    return {"cuda_ms": cuda, "wall_ms": wall, "cuda_stats_ms": stats(cuda), "wall_stats_ms": stats(wall),
            "allocated_peak_bytes": torch.cuda.max_memory_allocated(), "clocks_after": clocks()}


@contextmanager
def scopes(model):
    handles, stacks = [], {}
    for name, module in model.named_modules():
        def pre(m, args, name=name):
            ctx = torch.profiler.record_function("module::"+name+":"+type(m).__name__)
            ctx.__enter__()
            stacks.setdefault(id(m), []).append(ctx)
        def post(m, args, output):
            stacks[id(m)].pop().__exit__(None, None, None)
        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post, always_call=True))
    try:
        yield
    finally:
        for h in reversed(handles):
            h.remove()
        if any(stacks.values()):
            raise RuntimeError("Unclosed profiler scope")


def union(intervals):
    total, end = 0., -float("inf")
    for a, b in sorted(intervals):
        total += max(0., b-max(a, end))
        end = max(end, b)
    return total


def operator_group(op):
    op = op.lower()
    for label, names in (("convolution", ("conv",)), ("batch_norm", ("batch_norm",)),
                         ("activation", ("relu", "silu", "gelu")), ("upsample", ("upsample",)),
                         ("concat_stack", ("cat", "stack")), ("grid_sample", ("grid_sampler",)),
                         ("matmul_linear_attention", ("mm", "linear", "attention", "baddbmm")),
                         ("softmax", ("softmax",)), ("layer_norm", ("layer_norm",)),
                         ("copy_layout", ("copy", "contiguous", "clone"))):
        if any(n in op for n in names):
            return label
    return "other"


def summarize_trace(path):
    """Attribute each kernel once by runtime correlation and the innermost CPU/module range.

    Times are profiler observations. Inclusive parent CUDA sums are never added together.
    """
    events = json.loads(Path(path).read_text())["traceEvents"]
    ranges = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation")]
    runtimes = {e.get("args", {}).get("correlation"): e for e in events
                if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    kernels = [e for e in events if e.get("cat") == "kernel"]
    iterations = [e for e in ranges if e["name"].startswith("iteration::")]
    attributed, summary = [], []
    for iteration in iterations:
        records, launches = [], []
        for k in kernels:
            r = runtimes.get(k.get("args", {}).get("correlation"))
            if r is None or not (iteration["ts"] <= r["ts"] < iteration["ts"]+iteration["dur"]):
                continue
            parents = [p for p in ranges if p["tid"] == r["tid"] and p["ts"] <= r["ts"]
                       and p["ts"]+p["dur"] >= r["ts"]+r["dur"]-.01]
            ops = sorted([p for p in parents if p.get("cat") == "cpu_op"], key=lambda p: p["dur"])
            modules = sorted([p for p in parents if p["name"].startswith("module::")], key=lambda p: p["dur"])
            op = ops[0]["name"] if ops else "unattributed"
            module = modules[0]["name"][8:] if modules else "unattributed"
            top = next((p["name"][8:] for p in modules if p["name"][8:].split(":")[0].count(".") == 1), "unattributed")
            records.append({"iteration": iteration["name"], "kernel": k["name"], "op": op,
                            "group": operator_group(op), "module": module, "layer": top,
                            "gpu_start_us": k["ts"], "gpu_us": k["dur"], "launch_start_us": r["ts"],
                            "launch_us": r["dur"]})
            launches.append((r["ts"], r["ts"]+r["dur"]))
        if not records:
            raise RuntimeError("No correlated CUDA kernels; profiler result cannot support attribution")
        intervals = [(r["gpu_start_us"], r["gpu_start_us"]+r["gpu_us"]) for r in records]
        lo, hi = min(a for a,b in intervals), max(b for a,b in intervals)
        busy = union(intervals)
        unique_launches = sorted(set(launches))
        gaps = [max(0, b[0]-a[1]) for a,b in zip(unique_launches, unique_launches[1:])]
        gpu_gaps, right = [], lo
        for a,b in sorted(intervals):
            if a > right:
                gpu_gaps.append(a-right)
            right = max(right,b)
        summary.append({"iteration": iteration["name"], "cpu_forward_us": iteration["dur"],
                        "kernel_count": len(records), "kernel_sum_us": sum(r["gpu_us"] for r in records),
                        "gpu_span_us": hi-lo, "gpu_busy_union_us": busy, "gpu_idle_in_span_us": hi-lo-busy,
                        "gpu_idle_fraction": 1-busy/(hi-lo), "kernel_duration_us": stats([r["gpu_us"] for r in records]),
                        "cpu_launch_gap_us": stats(gaps) if gaps else None,
                        "gpu_gap_us": stats(gpu_gaps) if gpu_gaps else None})
        attributed.extend(records)
    if not summary:
        raise RuntimeError("Missing iteration scopes")
    tables = {}
    for key in ("op", "group", "layer", "module", "kernel"):
        buckets = {}
        for r in attributed:
            b = buckets.setdefault(r[key], {"name": r[key], "calls": 0, "cuda_us": 0.})
            b["calls"] += 1
            b["cuda_us"] += r["gpu_us"]
        tables[key] = sorted([{**b, "calls_per_forward": b["calls"]/len(summary),
                              "cuda_us_per_forward": b["cuda_us"]/len(summary)} for b in buckets.values()],
                             key=lambda b: -b["cuda_us"])
    # Small cross table answers Conv/BN/activation inside each top-level layer without overlap.
    buckets = {}
    for r in attributed:
        key = (r["layer"], r["group"])
        b = buckets.setdefault(key, {"layer": key[0], "group": key[1], "calls": 0, "cuda_us": 0.})
        b["calls"] += 1
        b["cuda_us"] += r["gpu_us"]
    tables["layer_group"] = [{**b, "calls_per_forward": b["calls"]/len(summary),
                              "cuda_us_per_forward": b["cuda_us"]/len(summary)} for b in buckets.values()]
    origin = min(r["gpu_start_us"] for r in attributed if r["iteration"] == summary[0]["iteration"])
    compact = [{**{k:r[k] for k in ("kernel", "op", "group", "layer", "module", "gpu_us", "launch_us")},
                "gpu_start_us": r["gpu_start_us"]-origin, "launch_start_us": r["launch_start_us"]-origin}
               for r in attributed if r["iteration"] == summary[0]["iteration"]]
    memcpy = [e for e in events if e.get("cat") in ("gpu_memcpy", "gpu_memset")]
    return {"iterations": summary, "tables": tables, "first_iteration_timeline": compact,
            "trace_memcpy_memset_count": len(memcpy),
            "trace": {"server_path": str(path), "bytes": Path(path).stat().st_size, "sha256": sha(path)}}


def profile(fn, model, path, *, labeled):
    from contextlib import nullcontext
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with scopes(model) if labeled else nullcontext():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=False, with_stack=False, profile_memory=False) as p:
            for i in range(3):
                with torch.profiler.record_function(f"iteration::{i}"):
                    fn()
                torch.cuda.synchronize()
                p.step()
    p.export_chrome_trace(str(path))
    return summarize_trace(path)


def run(args):
    configure()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run on host GPU, never substitute CPU timings")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    x, input_identity = load_input(args.data)
    report = {"schema": "5090.runtime-profile.v1", "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "source": source_identity(ROOT), "environment": environment("cuda:0"), "input": input_identity,
              "protocol": {"batch": 1, "shape": list(x.shape), "queries": 300, "points_per_level": [4,4,4],
                           "seed": 20261024, "warmup_initial": 150, "rounds": args.rounds, "warmup_block": 10,
                           "repeats_block": args.repeats, "profiler_iterations": 3,
                           "scope": "model forward only; resident GPU input/output, no H2D, preprocessing, decoding or NMS",
                           "timing": "CUDA events + perf_counter wall through end-event synchronize; profiler disabled",
                           "cudagraph": "static resident input/output; replay only, capture excluded"}, "models": {}}
    sources = [Path(__file__), *[ROOT/p for p in ("ultralytics/nn/tasks.py", "ultralytics/nn/modules/head.py",
               "ultralytics/nn/modules/transformer.py", "ultralytics/nn/modules/utils.py", "ultralytics/nn/modules/block.py",
               "ultralytics/nn/modules/conv.py", "ultralytics/nn/modules/rtdetr_na.py")]]
    report["source_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    for variant in args.models:
        path = args.runs / f"5090-rtdetr-r18-{variant}-150e-s0/weights/best.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        base = (saved.get("ema") or saved["model"]).float().eval().cuda()
        del saved
        reference = prediction(base(x)).clone()
        configs = [("eager_fp32", None, False, False), ("eager_fp16", torch.float16, False, False),
                   ("eager_bf16", torch.bfloat16, False, False), ("fused_fp32", None, True, False),
                   ("fused_fp16", torch.float16, True, False), ("fused_bf16", torch.bfloat16, True, False),
                   ("graph_fp32", None, False, True), ("graph_fused_fp32", None, True, True),
                   ("graph_fused_fp16", torch.float16, True, True)]
        conditions, functions, models, graphs = {}, {}, {}, []
        for name, dtype, fused, graph in configs:
            m = base if name in ("eager_fp32", "graph_fp32") else copy.deepcopy(base)
            info = {"status": "preparing", "dtype": str(dtype or torch.float32), "autocast": False, "fused": fused,
                    "cuda_graph": graph, "blocks": []}
            try:
                if fused:
                    info["fusion"] = fuse(m, full=True)
                if dtype:
                    m.to(dtype=dtype)
                xx = x.to(dtype=dtype or torch.float32)
                f = lambda m=m, xx=xx: m(xx)
                info["output_check_vs_fp32"] = comparison(reference, f())
                if graph:
                    try:
                        replay, g = capture(m, xx)
                        info["constant_creation"] = "native"
                    except RuntimeError as error:
                        if "CPU tensor is pinned" not in str(error):
                            raise
                        info["native_capture_failure"] = str(error)
                        # Explicit inference-only adapter: CPU-created AIFI positions and
                        # batch indices become CUDA constants. No learned parameters change.
                        def device_forward(value=xx, m=m):
                            with torch.device(value.device):
                                return m(value)
                        f = device_forward
                        info["constant_creation"] = "torch.device(cuda) context during eager/capture; GPU positions and batch indices"
                        info["adapted_output_vs_native_fp32"] = comparison(reference, f())
                        replay, g = capture(device_forward, xx)
                    eager_output = prediction(f()).clone()
                    eager_fn = f
                    f = replay
                    graphs.append(g)
                    info["graph_vs_own_eager"] = comparison(eager_output, f())
                    torch.testing.assert_close(prediction(f()), eager_output, atol=0, rtol=0)
                    # Replaying another real tensor verifies the graph reads its input, then restore it.
                    original = xx.clone()
                    xx.mul_(.75)
                    changed = prediction(eager_fn()).clone()
                    torch.testing.assert_close(prediction(f()), changed, atol=0, rtol=0)
                    xx.copy_(original)
                    info["changed_input_check"] = True
                conditions[name] = info
                functions[name], models[name] = f, m
                info["status"] = "ready"
                print(variant, name, "ready", flush=True)
            except Exception as e:
                info.update(status="failed", error=type(e).__name__+": "+str(e))
                conditions[name] = info
                print(variant, name, info["error"], flush=True)
        for f in functions.values():
            for _ in range(150):
                f()
        torch.cuda.synchronize()
        for round_index in range(args.rounds):
            order = list(functions)
            random.Random(20261025+round_index).shuffle(order)
            for name in order:
                block = measure(functions[name], 10, args.repeats)
                block["round"] = round_index+1
                conditions[name]["blocks"].append(block)
            print(variant, "timing round", round_index+1, "complete", flush=True)
        for name, f in functions.items():
            info = conditions[name]
            info["cuda_stats_ms"] = stats([v for b in info["blocks"] for v in b["cuda_ms"]])
            info["wall_stats_ms"] = stats([v for b in info["blocks"] for v in b["wall_ms"]])
            info["round_means_ms"] = [b["cuda_stats_ms"]["mean"] for b in info["blocks"]]
            info["status"] = "completed"
            print(variant, name, info["cuda_stats_ms"]["mean"], flush=True)
        # Two passes distinguish profiler overhead from the additional module-hook overhead.
        for name in ("eager_fp32", "fused_fp32", "graph_fp32", "graph_fused_fp16"):
            if name not in functions:
                continue
            info = conditions[name]
            for labeled in ([False, True] if name in ("eager_fp32", "fused_fp32") else [False]):
                suffix = "modules" if labeled else "plain"
                trace = out / f"{variant}-{name}-{suffix}.trace.json"
                info["profile_"+suffix] = profile(functions[name], models[name], trace, labeled=labeled)
                print("trace", trace.name, flush=True)
        # Probe stock fusion separately: the decoder Sequential projections are not folded by model.fuse().
        stock = copy.deepcopy(base)
        stock_info = fuse(stock, full=False)
        stock_info["output_check"] = comparison(reference, stock(x))
        stock_info["timing"] = measure(lambda: stock(x), 30, args.repeats)
        report["models"][variant] = {"checkpoint": {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size},
                                      "parameters": sum(p.numel() for p in base.parameters()),
                                      "conditions": conditions, "stock_fusion_probe": stock_info,
                                      "hooks_remaining": sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in base.modules())}
        write(out / "assessment.json", report)
        del functions, models, graphs, conditions, base, stock, reference, m, f
        torch.cuda.empty_cache()
    report["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    write(out / "assessment.json", report)
    print("OUTPUT", out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runs", type=Path, default=DEFAULT_RUN)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--models", nargs="+", choices=("c0", "c2"), default=["c0", "c2"])
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=50)
    args = p.parse_args()
    if args.rounds < 1 or args.repeats < 1:
        p.error("rounds/repeats must be positive")
    with torch.inference_mode():
        run(args)


if __name__ == "__main__":
    main()
