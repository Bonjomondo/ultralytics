"""Performance-only sampling/depth screen on real feature shapes; never trains.

Sampling replaces all three cross-attentions, including the p4 baseline, with
compatible modules under controlled initialization. Depth retains trained weights.
Each process handles one checkpoint to avoid cross-model process-order effects.
"""
import argparse
import copy
from datetime import datetime
import gc
from pathlib import Path
import random
import time
from zoneinfo import ZoneInfo

import torch
from torch import nn

from experiments.decoder_budget_5090.profile_runtime_5090 import (
    ROOT, DEFAULT_DATA, DEFAULT_RUN, capture, comparison, configure, environment,
    fuse, load_input, measure, prediction, sha, source_identity, stats, write,
)

CASES = ((300, 4), (300, 2), (200, 4), (200, 2), (300, 1))
SEED = 20261025
_TRT_LOGGER = None


def checkpoint_path(variant):
    return DEFAULT_RUN / f"5090-rtdetr-r18-{variant}-150e-s0/weights/best.pt"


def load_model(variant):
    saved = torch.load(checkpoint_path(variant), map_location="cpu", weights_only=False)
    return (saved.get("ema") or saved["model"]).float().eval().cuda()


def source_hashes(extra=()):
    paths = [Path(__file__), ROOT/"experiments/decoder_budget_5090/profile_runtime_5090.py",
             *[ROOT/p for p in ("ultralytics/nn/tasks.py", "ultralytics/nn/modules/head.py",
               "ultralytics/nn/modules/transformer.py", "ultralytics/nn/modules/utils.py")], *extra]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def replace_sampling(model, points, seed=SEED):
    """Change the actual linear dimensions and grid count, not a post-hoc mask."""
    from ultralytics.nn.modules.transformer import MSDeformAttn
    inventory = []
    for i, layer in enumerate(model.model[-1].decoder.layers):
        old = layer.cross_attn
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+i)
            baseline = MSDeformAttn(old.d_model, old.n_levels, old.n_heads, 4)
            # Native reset uses zero offset/weight matrices. Make these projections
            # input-dependent so an optimizing backend cannot benchmark constants.
            nn.init.normal_(baseline.sampling_offsets.weight, std=.01)
            nn.init.normal_(baseline.attention_weights.weight, std=.01)
            torch.manual_seed(seed+i)
            module = MSDeformAttn(old.d_model, old.n_levels, old.n_heads, points)
            # Value/output projections have identical bytes for every point budget.
            for name in ("value_proj", "output_proj"):
                getattr(module, name).load_state_dict(getattr(baseline, name).state_dict())
            with torch.no_grad():
                h,l,d = old.n_heads,old.n_levels,old.d_model
                module.sampling_offsets.weight.copy_(baseline.sampling_offsets.weight.view(h,l,4,2,d)[:,:,:points].reshape(-1,d))
                module.attention_weights.weight.copy_(baseline.attention_weights.weight.view(h,l,4,d)[:,:,:points].reshape(-1,d))
        module = module.to(device=next(old.parameters()).device, dtype=next(old.parameters()).dtype).eval()
        layer.cross_attn = module
        inventory.append({"layer": i+1, "points_per_level": [points]*old.n_levels,
                          "parameters": sum(p.numel() for p in module.parameters()),
                          "offset_outputs": module.sampling_offsets.out_features,
                          "weight_outputs": module.attention_weights.out_features})
    return inventory


class PredictionOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.network = model

    def forward(self, images):
        return prediction(self.network(images))


class AttentionOnly(nn.Module):
    def __init__(self, module, shapes):
        super().__init__()
        self.attention = module
        self.shapes = shapes

    def forward(self, query, reference, value):
        return self.attention(query, reference, value, self.shapes)


def attention_inputs(model, x):
    recorded = []
    def pre(module, args):
        if not recorded:
            recorded.append(([v.detach().clone() for v in args[:3]], copy.deepcopy(args[3])))
    handle = model.model[-1].decoder.layers[0].cross_attn.register_forward_pre_hook(pre)
    try:
        model(x)
    finally:
        handle.remove()
    inputs, shapes = recorded[0]
    return inputs, shapes


def count_execution(model, x, depth, queries, points):
    seen, handles = [], []
    for i, layer in enumerate(model.model[-1].decoder.layers):
        def hook(module, args, output, i=i):
            seen.append({"layer": i+1, "queries": output.shape[1], "points": module.cross_attn.n_points})
        handles.append(layer.register_forward_hook(hook))
    try:
        output = prediction(model(x))
        if output.shape != (1, queries, 14) or not torch.isfinite(output).all():
            raise RuntimeError("Unexpected model prediction shape/values")
    finally:
        for h in handles:
            h.remove()
    expected = [{"layer": i+1, "queries": queries, "points": points} for i in range(depth)]
    if seen != expected:
        raise RuntimeError(f"Budget did not reach native execution: {seen} != {expected}")
    return seen


def engine_runner(module, inputs, names, directory, fp16):
    """All standalone inputs remain graph inputs, preventing projection constant folding."""
    import tensorrt as trt
    import onnx
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/"model.onnx"
    fast = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        torch.onnx.export(module, tuple(inputs), str(path), input_names=names,
                          output_names=["predictions"], opset_version=17, dynamo=False, do_constant_folding=True)
    finally:
        torch.backends.mha.set_fastpath_enabled(fast)
    onnx.checker.check_model(str(path))
    graph = onnx.load(path).graph
    if {v.name for v in graph.input} != set(names):
        raise RuntimeError("Export dropped an input; constant-folded benchmark is invalid")
    global _TRT_LOGGER
    if _TRT_LOGGER is None:
        _TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    logger = _TRT_LOGGER
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(path)):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.builder_optimization_level = 3
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build failed")
    build_seconds = time.perf_counter()-start
    engine_path = directory/"model.engine"
    engine_path.write_bytes(bytes(serialized))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()
    for name, value in zip(names, inputs):
        if tuple(engine.get_tensor_shape(name)) != tuple(value.shape) or engine.get_tensor_dtype(name) != trt.float32:
            raise RuntimeError("Unexpected TensorRT input signature")
        context.set_tensor_address(name, value.data_ptr())
    output = torch.empty(tuple(engine.get_tensor_shape("predictions")), dtype=torch.float32, device="cuda")
    context.set_tensor_address("predictions", output.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream
    if stream == 0:
        raise RuntimeError("TensorRT screen requires a non-default shared stream")
    def run():
        if runtime is None or engine is None or not context.execute_async_v3(stream):
            raise RuntimeError("TensorRT enqueue failed")
        return output
    inspector = engine.create_engine_inspector()
    details = directory/"layers.json"
    details.write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    return run, {"tensorrt": trt.__version__, "fp16_allowed": fp16, "tf32": False,
                 "workspace_bytes": 2 << 30, "builder_optimization_level": 3, "build_seconds": build_seconds,
                 "input_names": names, "input_shapes": [list(x.shape) for x in inputs],
                 "engine_layers": engine.num_layers,
                 "server_artifacts": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}
                                      for p in (path,engine_path,details)]}


def ready_paths(module, inputs, names, directory, whole, backends):
    # The legacy exporter restores the wrapper's mode recursively. A newly
    # constructed nn.Module wrapper defaults to training even around an eval net.
    module.eval()
    functions, records, keep = {}, {}, [module, inputs]
    ref = module(*inputs).clone()
    for backend in backends:
        info = {"blocks": [], "status": "preparing"}
        try:
            if backend == "eager_fp32":
                fn = lambda: module(*inputs)
                info["fusion"] = "unfused trained structure" if whole else "not applicable"
            elif backend.startswith("graph"):
                model = copy.deepcopy(module)
                if whole:
                    info["fusion"] = fuse(model.network, full=True)
                dtype = torch.float16 if backend.endswith("fp16") else torch.float32
                model.to(dtype=dtype)
                xs = [x.to(dtype=dtype) for x in inputs]
                # GPU constant construction avoids native CPU->GPU graph capture failures.
                def adapted(_, model=model, xs=xs):
                    with torch.device(xs[0].device):
                        return model(*xs)
                expected = adapted(None).clone()
                fn, graph = capture(adapted, xs[0])
                torch.testing.assert_close(fn(),expected,atol=0,rtol=0)
                original = xs[0].clone()
                xs[0].mul_(.75)
                torch.testing.assert_close(fn(),adapted(None),atol=0,rtol=0)
                xs[0].copy_(original)
                info["graph_exact_own_eager"] = True
                info["changed_input_check"] = True
                keep.extend([model,xs,graph])
            elif backend.startswith("trt"):
                model = copy.deepcopy(module)
                if whole:
                    info["fusion"] = fuse(model.network,full=True)
                fn, info["engine"] = engine_runner(model,inputs,names,directory/backend,backend.endswith("fp16"))
                # Changed input is checked against corresponding PyTorch, not just for finite output.
                original = inputs[0].clone()
                inputs[0].mul_(.75)
                info["changed_input_output"] = comparison(model(*inputs),fn())
                inputs[0].copy_(original)
                keep.append(model)
            else:
                raise ValueError(backend)
            info["output_check"] = comparison(ref,fn())
            functions[backend] = fn
            info["status"] = "ready"
            print(directory.name,backend,"ready",flush=True)
        except Exception as error:
            info.update(status="failed",error=type(error).__name__+": "+str(error))
            print(directory.name,backend,info["error"],flush=True)
        records[backend] = info
    return functions, records, keep


def run(args):
    configure()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; do not substitute CPU results")
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    x, input_identity = load_input(DEFAULT_DATA)
    base = load_model(args.variant)
    actual_inputs, shapes = attention_inputs(base,x)
    checkpoint = checkpoint_path(args.variant)
    record = {"schema":"5090.decoder-performance.v1", "kind":args.kind,"variant":args.variant,
              "started_at":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "source":source_identity(ROOT), "source_sha256":source_hashes(),
              "environment":environment("cuda:0"),"input":input_identity,
              "checkpoint":{"path":str(checkpoint),"bytes":checkpoint.stat().st_size,"sha256":sha(checkpoint)},
              "protocol":{"rounds":args.rounds,"repeats_block":args.repeats,"warmup_initial":150,"warmup_block":10,
                          "seed":SEED,"batch":1,"size":640,"feature_shapes":shapes,
                          "scope":"resident CUDA inputs/outputs; model forward only; no data I/O/pre/postprocess",
                          "stream":"shared non-default CUDA stream", "sampling_accuracy":"not evaluated; compatible random cross-attention in ALL point conditions",
                          "depth_weights":"original checkpoint; native eval_idx exits after specified layer",
                          "initialization":"module seed=20261025+layer_index; common p4 value/output weights; offset/attention matrices N(0,.01) with per-head/level point prefixes; native ring offset bias and zero attention bias",
                          "precision":"all FP32 inputs/params except graph_fp16; TRT fp16 allows mixed precision with FP32 I/O; TF32 disabled"},
              "conditions":{},"performance_groups":{}}
    functions, groups, keep = {}, {}, []
    cases = [(f"q{q}-p{p}",q,p,3) for q,p in CASES] if args.kind == "sampling" else [(f"d{d}",300,4,d) for d in (1,2,3)]
    if args.cases:
        if not set(args.cases).issubset({c[0] for c in cases}):
            raise ValueError("Unknown condition in --cases")
        cases = [c for c in cases if c[0] in args.cases]
    for case, queries, points, depth in cases:
        model = copy.deepcopy(base)
        model.model[-1].num_queries = queries
        model.model[-1].decoder.eval_idx = depth-1
        inventory = replace_sampling(model,points) if args.kind == "sampling" else []
        execution = count_execution(model,x,depth,queries,points)
        record["conditions"][case] = {"queries":queries,"points_per_level":[points]*3,"depth":depth,
                                      "executed":execution,"attention_inventory":inventory,
                                      "parameters":sum(p.numel() for p in model.parameters())}
        workloads = [("model",PredictionOnly(model),[x],["images"])]
        if args.kind == "sampling":
            q,reference,value = actual_inputs
            workloads.append(("attention", AttentionOnly(copy.deepcopy(model.model[-1].decoder.layers[0].cross_attn),shapes),
                              [q[:,:queries].contiguous(),reference[:,:queries].contiguous(),value], ["query","reference","value"]))
        for workload,module,inputs,names in workloads:
            directory = output/f"{case}-{workload}"
            fs,rs,resources = ready_paths(module,inputs,names,directory,workload=="model",args.backends)
            key = case+"/"+workload
            record["performance_groups"][key] = {"case":case,"workload":workload,
                                               "input_shapes":[list(t.shape) for t in inputs],"backends":rs}
            for backend,fn in fs.items():
                functions[key+"/"+backend] = fn
                groups[key+"/"+backend] = rs[backend]
            keep.extend(resources)
        write(output/"progress.json",record)
    # Initialization/build/capture and profiler are excluded from all measured blocks.
    for fn in functions.values():
        for _ in range(150):
            fn()
    torch.cuda.synchronize()
    for r in range(args.rounds):
        order = list(functions)
        random.Random(SEED+r).shuffle(order)
        for key in order:
            block = measure(functions[key],10,args.repeats)
            block["round"] = r+1
            groups[key]["blocks"].append(block)
        print(args.variant,args.kind,"timing round",r+1,flush=True)
        write(output/"progress.json",record)
    for key,info in groups.items():
        info["cuda_stats_ms"] = stats([v for b in info["blocks"] for v in b["cuda_ms"]])
        info["wall_stats_ms"] = stats([v for b in info["blocks"] for v in b["wall_ms"]])
        info["status"] = "completed"
        print(key,info["cuda_stats_ms"]["mean"],flush=True)
    record["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    if any(m._forward_hooks or m._forward_pre_hooks for m in base.modules()):
        raise RuntimeError("Hooks leaked")
    record["hooks_remaining"] = 0
    write(output/"assessment.json",record)
    print("OUTPUT",output,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind",choices=["sampling","depth"])
    p.add_argument("--variant",choices=["c0","c2"],required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--rounds",type=int,default=10)
    p.add_argument("--repeats",type=int,default=50)
    p.add_argument("--cases",nargs="+",help="Optional engineering subset; omit for the complete formal screen")
    p.add_argument("--backends",nargs="+",choices=["eager_fp32","graph_fp32","graph_fp16","trt_fp32","trt_fp16"],
                   default=["eager_fp32","graph_fp32","graph_fp16","trt_fp32","trt_fp16"])
    args = p.parse_args()
    if args.rounds < 2 or args.repeats < 1:
        p.error("Require rounds>=2 and repeats>=1")
    with torch.inference_mode():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(args)
        stream.synchronize()


if __name__ == "__main__":
    main()
