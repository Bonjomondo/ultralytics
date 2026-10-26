"""Randomized four-arm native FP32 timing, before training or with four own checkpoints."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from experiments.decoder_budget_5090.train_5090 import now, runtime_environment, sha256_file, write_atomic
from experiments.p2_p5_la_5090.model import MODELS, build_model, inspect_model
from experiments.p2_p5_la_5090.train import configure, source_hashes
from experiments.r18_p2_sfi.train import tensor_digest


def stats(values):
    return {"mean_ms": float(np.mean(values)), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "samples": len(values)}


def attach_events(model, events):
    return [model.model[4].register_forward_hook(lambda m, a, y: events[1].record()),
            model.model[6].register_forward_pre_hook(lambda m, a: events[2].record()),
            model.model[6].register_forward_hook(lambda m, a, y: events[3].record()),
            model.model[23].register_forward_hook(lambda m, a, y: events[4].record())]


def benchmark(args):
    if args.rounds < 5 or args.samples < 1:
        raise ValueError("Require at least five paired rounds and positive samples")
    configure()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoints = dict(zip(MODELS, args.checkpoints)) if args.checkpoints else {}
    record = {"schema": "5090.p2-p5-la.benchmark.v1", "status": "running", "started_utc": now(),
              "role": "trained_checkpoint_comparison" if checkpoints else "untrained_architecture_screen_no_AP",
              "environment": runtime_environment(), "source_sha256": source_hashes(),
              "protocol": {"dtype": "float32", "fused": False, "batch": 1, "imgsz": 640,
                           "rounds": args.rounds, "samples_per_round": args.samples, "initial_warmup": 150,
                           "block_warmup": 10, "order_seed": 20261026,
                           "scope": "GPU-resident forward; includes CUDA-stream idle gaps; excludes transfers/pre/postprocess",
                           "p5_is_subset_of_neck_encoder": True}, "models": {}, "round_order": []}
    if args.image:
        import cv2
        image = cv2.imread(str(args.image))
        if image is None:
            raise ValueError("Cannot decode benchmark image")
        x = torch.from_numpy(cv2.resize(image, (640, 640))[..., ::-1].copy()).permute(2, 0, 1)[None].cuda().float() / 255
        record["input"] = {"path": str(args.image.resolve()), "sha256": sha256_file(args.image)}
    else:
        torch.manual_seed(19)
        x = torch.rand(1, 3, 640, 640, device="cuda")
        record["input"] = {"kind": "synthetic_random", "seed": 19, "interpretation": "architecture timing only"}
    models = {}
    for variant in MODELS:
        if variant in checkpoints:
            path = checkpoints[variant].resolve()
            saved = torch.load(path, map_location="cpu", weights_only=False)
            model = (saved.get("ema") or saved["model"]).float()
            identity = {"path": str(path), "sha256": sha256_file(path)}
        else:
            model = build_model(str(MODELS[variant]))
            identity = {"initialization": model.ablation_initialization}
        model = model.eval().cuda()
        if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
            raise RuntimeError("Benchmark input contains observer hooks")
        models[variant] = model
        record["models"][variant] = {"architecture": inspect_model(model, variant), "identity": identity,
                                     "parameters_unfused": sum(p.numel() for p in model.parameters()),
                                     "state_sha256_before": tensor_digest(model.state_dict()), "blocks": []}
        for _ in range(150):
            model(x)
    torch.cuda.synchronize()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
    for round_index in range(args.rounds):
        order = [(v, mode) for v in MODELS for mode in ("bare", "staged")]
        random.Random(20261026 + round_index).shuffle(order)
        record["round_order"].append(order)
        for variant, mode in order:
            model = models[variant]
            native = model(x)[0]
            handles = attach_events(model, events) if mode == "staged" else []
            values = {"whole": []}
            if mode == "staged":
                values.update({k: [] for k in ("backbone", "p5", "neck_encoder", "head_decoder")})
            try:
                torch.testing.assert_close(model(x)[0], native, atol=0, rtol=0)
                for _ in range(10):
                    model(x)
                torch.cuda.synchronize()
                for _ in range(args.samples):
                    events[0].record()
                    model(x)
                    events[5].record()
                    events[5].synchronize()
                    whole = events[0].elapsed_time(events[5])
                    values["whole"].append(whole)
                    if mode == "staged":
                        durations = {"backbone": events[0].elapsed_time(events[1]),
                                     "p5": events[2].elapsed_time(events[3]),
                                     "neck_encoder": events[1].elapsed_time(events[4]),
                                     "head_decoder": events[4].elapsed_time(events[5])}
                        if abs(sum(durations[k] for k in ("backbone", "neck_encoder", "head_decoder")) - whole) > .02:
                            raise RuntimeError("Stage accounting failed")
                        for key, value in durations.items():
                            values[key].append(value)
            finally:
                for handle in reversed(handles):
                    handle.remove()
            record["models"][variant]["blocks"].append({"round": round_index + 1, "mode": mode, "timings_ms": values})
        write_atomic(args.output / "progress.json", record)
        print("completed paired round", round_index + 1, flush=True)
    for variant, model in models.items():
        item = record["models"][variant]
        if tensor_digest(model.state_dict()) != item["state_sha256_before"]:
            raise RuntimeError("Benchmark mutated model state")
        if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
            raise RuntimeError("Benchmark hooks leaked")
        item.update(model_state_unchanged=True, hooks_remaining=0, output_exact_with_hooks=True)
        item["summary"] = {mode: {key: stats([v for b in item["blocks"] if b["mode"] == mode
                                               for v in b["timings_ms"][key]])
                                  for key in (("whole",) if mode == "bare" else
                                              ("whole", "backbone", "p5", "neck_encoder", "head_decoder"))}
                           for mode in ("bare", "staged")}
        item["instrumentation_overhead_percent"] = 100 * (
            item["summary"]["staged"]["whole"]["mean_ms"] / item["summary"]["bare"]["whole"]["mean_ms"] - 1)
        item["gpu_fps_from_bare_mean"] = 1000 / item["summary"]["bare"]["whole"]["mean_ms"]
    record.update(status="completed", completed_utc=now())
    write_atomic(args.output / "benchmark.json", record)


if __name__ == "__main__":
    import fcntl
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image", type=Path)
    p.add_argument("--checkpoints", type=Path, nargs=4, metavar=("C0", "C2", "LA", "C2_LA"))
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--samples", type=int, default=100)
    args = p.parse_args()
    with open("/tmp/zhj-uav-5090-gpu0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with torch.inference_mode():
            benchmark(args)
