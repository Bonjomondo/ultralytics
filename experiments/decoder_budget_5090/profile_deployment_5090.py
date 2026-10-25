"""Export fixed-shape ONNX and benchmark ONNX CUDA / TensorRT with resident GPU I/O.

All exported models and engines remain under ignored runs/. This is a performance
and output-smoke experiment, not a low-precision AP evaluation or a production server.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

from experiments.decoder_budget_5090.profile_runtime_5090 import (
    ROOT, DEFAULT_DATA, DEFAULT_RUN, comparison, configure, environment, fuse,
    load_input, measure, prediction, profile, sha, source_identity, stats, write,
)


def ort_runner(path, x, shape):
    import onnxruntime as ort
    ort.preload_dlls()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CUDA provider unavailable; CPU substitution forbidden")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [("CUDAExecutionProvider", {"device_id": 0, "user_compute_stream": str(torch.cuda.current_stream().cuda_stream),
                                            "use_tf32": 0, "cudnn_conv_algo_search": "HEURISTIC"})]
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("ONNX Runtime silently fell back from CUDA")
    output = torch.empty(shape, device="cuda", dtype=torch.float32)
    binding = session.io_binding()
    binding.bind_input("images", "cuda", 0, np.float32, tuple(x.shape), x.data_ptr())
    binding.bind_output("predictions", "cuda", 0, np.float32, tuple(output.shape), output.data_ptr())
    def run():
        session.run_with_iobinding(binding)
        return output
    return run, {"version": ort.__version__, "providers": session.get_providers(),
                 "provider_options": session.get_provider_options(), "io_binding": "resident torch CUDA pointers",
                 "cpu_fallback": "CPU EP allowed for graph/shape operations; CUDA EP is required and first"}


def trt_runner(path, x, out, fp16):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
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
    t = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT builder returned None; see server log")
    build_seconds = time.perf_counter()-t
    engine_path = out / ("fp16.engine" if fp16 else "fp32.engine")
    engine_path.write_bytes(bytes(serialized))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()
    output = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            assert name == "images" and tuple(engine.get_tensor_shape(name)) == tuple(x.shape)
            assert engine.get_tensor_dtype(name) == trt.float32
            context.set_tensor_address(name, x.data_ptr())
        else:
            assert name == "predictions" and engine.get_tensor_dtype(name) == trt.float32
            output = torch.empty(tuple(engine.get_tensor_shape(name)), device="cuda", dtype=torch.float32)
            context.set_tensor_address(name, output.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream
    def run():
        # The closure retains runtime/engine/context/output and uses the timing event's stream.
        if runtime is None or engine is None or not context.execute_async_v3(stream):
            raise RuntimeError("TensorRT enqueue failed")
        return output
    inspector = engine.create_engine_inspector()
    info_path = out / ("fp16.layers.json" if fp16 else "fp32.layers.json")
    info_path.write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    return run, {"version": trt.__version__, "fp16_allowed": fp16, "tf32": False,
                 "note": "FP16 flag permits mixed FP16/FP32 internal tactics; FP32 I/O",
                 "workspace_limit_bytes": 2 << 30, "builder_optimization_level": 3,
                 "build_seconds": build_seconds, "engine_layers": engine.num_layers,
                 "engine": {"server_path": str(engine_path), "bytes": engine_path.stat().st_size, "sha256": sha(engine_path)},
                 "layer_inspector": {"server_path": str(info_path), "bytes": info_path.stat().st_size, "sha256": sha(info_path)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--runs", default=DEFAULT_RUN, type=Path)
    p.add_argument("--data", default=DEFAULT_DATA, type=Path)
    p.add_argument("--models", nargs="+", default=["c0", "c2"], choices=["c0", "c2"])
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=50)
    args = p.parse_args()
    configure()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    x, identity = load_input(args.data)
    report = {"schema": "5090.deployment-profile.v1", "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "source": source_identity(ROOT), "environment": environment("cuda:0"), "input": identity,
              "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(__file__),
                                "experiments/decoder_budget_5090/profile_runtime_5090.py": sha(ROOT/"experiments/decoder_budget_5090/profile_runtime_5090.py")},
              "protocol": {"scope": "batch1 640 square 300Q [4,4,4], resident CUDA I/O, model predictions only",
                           "rounds": args.rounds, "repeats_block": args.repeats, "warmup_block": 10,
                           "warmup_initial": 150, "timing": "CUDA events and synchronized wall; build/export excluded",
                           "stream": "dedicated non-default torch CUDA stream shared by events, ONNX and TensorRT",
                           "accuracy": "finite/shape/output differences only; no AP claim"}, "models": {}}
    for variant in args.models:
        directory = out / variant
        directory.mkdir()
        checkpoint = args.runs / f"5090-rtdetr-r18-{variant}-150e-s0/weights/best.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = (saved.get("ema") or saved["model"]).float().eval().cuda()
        del saved
        reference = prediction(model(x)).clone()
        fusion = fuse(model, full=True)
        native = lambda: prediction(model(x))
        conditions = {"eager_fused_fp32": {"status": "ready", "blocks": [], "output_check": comparison(reference, native())}}
        functions = {"eager_fused_fp32": native}
        onnx_path = directory / "model.onnx"
        export = {"status": "preparing", "opset": 17, "dynamo": False, "fixed_shape": True,
                  "output": "final sigmoid boxes/scores [1,300,14]; no postprocess"}
        try:
            model.model[-1].export = True
            # Legacy exporter cannot lower the native MHA fastpath. Only export disables it.
            fastpath = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
            try:
                torch.onnx.export(model, x, str(onnx_path), input_names=["images"], output_names=["predictions"],
                                  opset_version=17, dynamo=False, do_constant_folding=True)
            finally:
                torch.backends.mha.set_fastpath_enabled(fastpath)
            import onnx
            onnx.checker.check_model(str(onnx_path))
            export.update(status="completed", onnx_version=onnx.__version__, server_path=str(onnx_path),
                          bytes=onnx_path.stat().st_size, sha256=sha(onnx_path),
                          nodes=len(onnx.load(onnx_path).graph.node), mha_fastpath_disabled_during_export=True)
            print(variant, "ONNX exported", flush=True)
            for name in ("onnx_cuda_fp32", "trt_fp32", "trt_fp16"):
                info = {"status": "preparing", "blocks": []}
                try:
                    if name.startswith("onnx"):
                        f, backend = ort_runner(onnx_path, x, reference.shape)
                    else:
                        f, backend = trt_runner(onnx_path, x, directory, name == "trt_fp16")
                    info["backend"] = backend
                    info["output_check"] = comparison(reference, f())
                    # Check that pointer-bound backends actually consume changed input.
                    original = x.clone()
                    x.mul_(.75)
                    changed = prediction(model(x)).clone()
                    info["changed_input_output_check"] = comparison(changed, f())
                    x.copy_(original)
                    functions[name] = f
                    info["status"] = "ready"
                    print(variant, name, "ready", flush=True)
                except Exception as e:
                    info.update(status="failed", error=type(e).__name__+": "+str(e))
                    print(variant, name, info["error"], flush=True)
                conditions[name] = info
        except Exception as e:
            export.update(status="failed", error=type(e).__name__+": "+str(e))
            print(variant, "ONNX", export["error"], flush=True)
        for f in functions.values():
            for _ in range(150):
                f()
        torch.cuda.synchronize()
        for r in range(args.rounds):
            order = list(functions)
            random.Random(20261025+r).shuffle(order)
            for name in order:
                b = measure(functions[name], 10, args.repeats)
                b["round"] = r+1
                conditions[name]["blocks"].append(b)
            print(variant, "deployment round", r+1, flush=True)
        for name in functions:
            info = conditions[name]
            info.update(status="completed", cuda_stats_ms=stats([v for b in info["blocks"] for v in b["cuda_ms"]]),
                        wall_stats_ms=stats([v for b in info["blocks"] for v in b["wall_ms"]]))
            print(variant, name, info["cuda_stats_ms"]["mean"], flush=True)
            info["profile_plain"] = profile(functions[name], model, directory / (name+".trace.json"), labeled=False)
        report["models"][variant] = {"checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)},
                                      "export": export, "fusion": fusion, "conditions": conditions}
        write(out / "assessment.json", report)
        del functions, conditions, model, reference, f, native
        torch.cuda.empty_cache()
    report["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    write(out / "assessment.json", report)
    print("OUTPUT", out, flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            main()
        stream.synchronize()
