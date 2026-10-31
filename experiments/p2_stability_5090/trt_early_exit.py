"""Full-val FP16 AP of the existing C2 D2/D3 engines, without rebuilding them."""
import argparse
import fcntl
from pathlib import Path
from types import SimpleNamespace

from experiments.p2_stability_5090.same_gt import DATA, DEPTH, OLD, ROOT, read, sha, table, write


def run(output):
    import torch
    from experiments.p2_p5_la_5090.evaluate_trt_fp16_5090 import evaluate
    perf_path = ROOT / "runs/diagnostics/5090-depth-perf-c2-20261025-final/assessment.json"
    perf = read(perf_path)
    native = read(DEPTH / "summary.json")
    training = OLD / "5090-rtdetr-r18-c2-150e-s0/experiment_5090.json"
    checkpoint = perf["checkpoint"]
    if (perf["kind"] != "depth" or perf["variant"] != "c2"
            or checkpoint["sha256"] != read(training)["checkpoints"]["best"]["sha256"]
            or sha(checkpoint["path"]) != checkpoint["sha256"]):
        raise ValueError("Require the original C2 three-layer checkpoint")
    conditions = {}
    aliases = {"c2": "d3", "c2-la": "d2"}
    for alias, actual in aliases.items():
        depth = int(actual[1:])
        if len(perf["conditions"][actual]["executed"]) != depth:
            raise ValueError("Early-exit execution receipt mismatch")
        backend = perf["performance_groups"][actual + "/model"]["backends"]["trt_fp16"]
        conditions[alias + "/trt_fp16"] = {"status": backend["status"], "engine": backend["engine"],
                                              "checkpoint": checkpoint}
    output.mkdir(parents=True, exist_ok=False)
    write(output / "benchmark-view.json", {"status": "completed", "input": perf["input"], "conditions": conditions,
          "actual_depth_by_alias": aliases, "source_receipt": {"path": str(perf_path), "sha256": sha(perf_path)}})
    write(output / "native-view.json", {"scope": "full_val", "protocol": {
          "image_ids": native["formal_protocol"]["image_ids"],
          "annotations_sha256": native["formal_protocol"]["annotation_sha256"]}})
    with torch.inference_mode():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            evaluate(SimpleNamespace(benchmark=output / "benchmark-view.json", training=training,
                     native_evaluation=output / "native-view.json", annotations=DATA / "annotations/VisDrone2019-DET_val_coco.json",
                     images=DATA / "images/val", output=output / "fp16"))
        stream.synchronize()
    result = read(output / "fp16/summary.json")
    result["actual_depth_by_alias"] = aliases
    result["models"] = {aliases[k]: v for k, v in result["models"].items()}
    result["source_receipt"] = {"path": str(perf_path), "sha256": sha(perf_path)}
    result["limitation"] = "AP measured now using the exact previously timed engines. Latency is historical 5090 GPU-resident forward, not new timing or Orin."
    write(output / "summary.json", result)
    rows = []
    for depth in ("d2", "d3"):
        rows.append({"depth": depth, "native_fp32_AP": 100 * read(DEPTH / (depth + "-coco.json"))["metrics"]["AP"],
                     "trt_fp16_AP": 100 * result["models"][depth]["metrics"]["AP"],
                     "trt_fp16_APs": 100 * result["models"][depth]["metrics"]["APs"],
                     "historical_same_engine_forward_ms": perf["performance_groups"][depth + "/model"]["backends"]["trt_fp16"]["cuda_stats_ms"]["mean"]})
    table(output / "comparison.csv", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with open("/tmp/zhj-uav-5090-gpu0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.output.resolve())
