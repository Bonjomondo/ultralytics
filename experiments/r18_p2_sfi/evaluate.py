"""Evaluate a completed SFI arm with the R0 native FP32/COCO protocol and GPU timing."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl")


class ConfidenceAuditValidator(RTDETRValidator):
    """Compare legacy and fixed masks using query identities on every validation image."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_audit = []
        self.pending_audit = []

    def postprocess(self, preds):
        raw = preds[0] if isinstance(preds, (list, tuple)) else preds
        self.pending_audit = []
        for scores in raw[..., 4:].amax(-1):
            order = scores.argsort(descending=True)
            old_mask = scores > self.args.conf
            fixed_mask = scores[order] > self.args.conf
            self.pending_audit.append({
                "queries": scores.numel(), "filtered_queries": int((~fixed_mask).sum()),
                "changed": bool((old_mask != fixed_mask).any()),
                "wrongly_removed_above_threshold": int((fixed_mask & ~old_mask).sum()),
                "wrongly_retained_below_threshold": int((old_mask & ~fixed_mask).sum()),
                "min_score": float(scores.min()), "max_score": float(scores.max()),
            })
        return super().postprocess(preds)

    def update_metrics(self, preds, batch):
        if len(self.pending_audit) != len(batch["im_file"]):
            raise RuntimeError("Confidence audit batch coverage mismatch")
        for record, path in zip(self.pending_audit, batch["im_file"]):
            self.filter_audit.append({"file_name": Path(path).name, **record})
        return super().update_metrics(preds, batch)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class ValidationHealth:
    """Observe every real validation image; skip the AutoBackend warmup forward."""

    def __init__(self, wrapper):
        self.capture = False
        self.values = {}
        self.handles = []
        head = wrapper.model.model[-1]
        for name, mlp in [("encoder", head.enc_bbox_head), *[
            (f"d{i + 1}", layer) for i, layer in enumerate(head.dec_bbox_head)]]:
            def hidden(module, inputs, label=name):
                if self.capture:
                    value = inputs[0].float()
                    self.values.setdefault(f"{label}_hidden_active_ratio", []).extend(
                        (value > 0).float().mean((1, 2)).cpu().tolist())

            def delta(module, inputs, output, label=name):
                if self.capture:
                    if not torch.isfinite(output).all():
                        raise FloatingPointError("Nonfinite validation bbox update")
                    self.values.setdefault(f"{label}_delta_query_variance", []).extend(
                        output.float().var(1, unbiased=False).mean(1).cpu().tolist())

            self.handles.extend((mlp.layers[-1].register_forward_pre_hook(hidden),
                                 mlp.register_forward_hook(delta)))
        wrapper.add_callback("on_val_batch_start", lambda validator: setattr(self, "capture", True))
        wrapper.add_callback("on_val_batch_end", lambda validator: setattr(self, "capture", False))

    def finish(self):
        for handle in self.handles:
            handle.remove()
        return {name: {"images": len(values), "mean": float(np.mean(values)),
                       "min": min(values), "max": max(values),
                       "zero_images": sum(value == 0 for value in values)}
                for name, values in self.values.items()}


def benchmark(weight):
    """Batch-one fused PyTorch FP32 GPU forward; excludes host transfers and pre/postprocessing."""
    model = RTDETR(str(weight)).model.float().cuda().eval()
    params = sum(p.numel() for p in model.parameters())
    gflops = get_flops(model, imgsz=640)
    model.fuse(verbose=False)
    torch.manual_seed(19)
    x = torch.rand(1, 3, 640, 640, device="cuda")
    with torch.inference_mode():
        for _ in range(50):
            model(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(200):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(x)
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
        peak = torch.cuda.max_memory_allocated() / 2**20
    return {"parameters_unfused": params, "gflops_thop_640": gflops,
            "scope": "fused PyTorch FP32 batch=1 640x640 GPU forward only; no transfer/pre/postprocess",
            "warmup": 50, "repetitions": 200, "latency_ms": timings,
            "mean_ms": float(np.mean(timings)), "median_ms": float(np.median(timings)),
            "p95_ms": float(np.percentile(timings, 95)), "fps_from_mean": 1000 / float(np.mean(timings)),
            "peak_allocated_mib": peak, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("c0", "c1", "c2", "c3"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--expected-epochs", type=int, default=150)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=Path, help="Override best.pt, e.g. compare last.pt")
    parser.add_argument("--allow-partial", action="store_true", help="Evaluate an early-stopped run with <= expected epochs")
    args = parser.parse_args()
    OUTPUT = args.output_dir.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    with (args.run / "results.csv").open() as file:
        rows = list(csv.DictReader(file))
    actual_epochs = int(rows[-1]["epoch"])
    if len(rows) != actual_epochs or (actual_epochs != args.expected_epochs and not (
            args.allow_partial and 0 < actual_epochs <= args.expected_epochs)):
        raise RuntimeError("Evaluation requires the requested complete training budget")
    weight = args.checkpoint or args.run / "weights/best.pt"
    annotation = ROOT.parent / "datasets/VisDrone2019-DET-YOLO/annotations/VisDrone2019-DET_val_coco.json"
    historical = json.loads((ROOT / "experiments/r18_na_ablation/completed_pair/r0_formal_metrics.json").read_text())
    if digest(annotation) != historical["protocol"]["annotation_sha256"]:
        raise RuntimeError("GT differs from the audited R0 baseline")
    model = RTDETR(str(weight))
    health = ValidationHealth(model)
    audit = []
    model.add_callback("on_val_end", lambda validator: audit.extend(validator.filter_audit))
    result = model.val(validator=ConfidenceAuditValidator, conf=0.001,
                       data=str(ROOT / "configs/visdrone-local.yaml"), device="0", imgsz=640,
                       batch=2, workers=0, half=False, plots=False, verbose=False, save_json=True,
                       max_det=300, project=str(args.run), name="formal_native", exist_ok=False)
    validation_health = health.finish()
    if len(validation_health) != 8 or any(value["images"] != 548 for value in validation_health.values()):
        raise RuntimeError("Full-val regression health observation missed images or decoder stages")
    prediction_path = Path(result.save_dir) / "predictions.json"
    predictions = json.loads(prediction_path.read_text())
    if len(audit) != 548 or len({row["file_name"] for row in audit}) != 548:
        raise RuntimeError("Confidence audit did not cover 548 unique validation images")
    audit_path = Path(result.save_dir) / "confidence_filter_audit.json"
    write(audit_path, audit)
    gt = COCO(str(annotation))
    images = {Path(image["file_name"]).name: image["id"] for image in gt.dataset["images"]}
    for prediction in predictions:
        prediction["image_id"] = images[prediction["file_name"]]
    evaluator = COCOeval(gt, gt.loadRes(predictions), "bbox")
    evaluator.params.imgIds = sorted(gt.getImgIds())
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    metrics = dict(zip(METRICS, map(float, evaluator.stats)))
    protocol = {"iou_type": "bbox", "image_ids": [int(i) for i in evaluator.params.imgIds],
                "max_dets": [1, 10, 100], "engine": "pycocotools.COCOeval",
                "engine_version": version("pycocotools"), "iou_thresholds": evaluator.params.iouThrs.tolist(),
                "area": "all", "annotation_sha256": digest(annotation),
                "category_ids": [int(i) for i in sorted(gt.cats)]}
    if protocol != historical["protocol"]:
        raise RuntimeError("Evaluation protocol differs from audited R0")
    per_class = {}
    for index, category in enumerate(evaluator.params.catIds):
        per_class[gt.cats[category]["name"]] = {}
        for area, label in enumerate(("AP", "APs", "APm", "APl")):
            values = evaluator.eval["precision"][:, :, index, area, -1]
            values = values[values > -1]
            per_class[gt.cats[category]["name"]][label] = float(values.mean()) if values.size else None
    record = {"variant": args.variant, "training_epochs": actual_epochs,
              "finished_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint": str(weight), "checkpoint_sha256": digest(weight), "protocol": protocol,
              "native_internal_metrics": {key: float(value) for key, value in result.results_dict.items()},
              "predictions_path": str(prediction_path), "predictions_file_sha256": digest(prediction_path),
              "metrics": metrics, "per_class": per_class, "benchmark": benchmark(weight)}
    record["validation_regression_health"] = validation_health
    record["confidence_filter_audit"] = {
        "conf": 0.001, "images": len(audit),
        **{key: sum(row[key] for row in audit) for key in (
            "queries", "filtered_queries", "changed", "wrongly_removed_above_threshold",
            "wrongly_retained_below_threshold")},
        "min_score": min(row["min_score"] for row in audit),
        "per_image_path": str(audit_path), "per_image_sha256": digest(audit_path),
        "scope": "raw best-checkpoint fused FP32 validation queries; legacy mask versus fixed sorted mask",
    }
    record["postprocess"] = "sorted confidence mask fixed; val.py SHA256=" + digest(ROOT / "ultralytics/models/rtdetr/val.py")
    write(OUTPUT / f"{args.variant}_formal_metrics.json", record)
    summarize()


def summarize():
    records = [json.loads(path.read_text()) for variant in ("c0", "c1", "c2", "c3")
               if (path := OUTPUT / f"{variant}_formal_metrics.json").exists()]
    baseline = next((record for record in records if record["variant"] == "c0"), None)
    table = []
    for record in records:
        row = {"variant": record["variant"], **{name: value * 100 for name, value in record["metrics"].items()},
               **{key: record["benchmark"][key] for key in ("parameters_unfused", "gflops_thop_640", "median_ms", "p95_ms")}}
        if baseline:
            delta = {name: (record["metrics"][name] - baseline["metrics"][name]) * 100 for name in METRICS}
            row.update({f"delta_{name}_pp": value for name, value in delta.items()})
            row["screen_pass"] = (record["variant"] != "c0" and delta["AP"] >= 0.3 and delta["APs"] >= 0.5
                                   and all(delta[key] >= -0.5 for key in ("AP75", "APm", "APl"))
                                   and record["benchmark"]["median_ms"] <= baseline["benchmark"]["median_ms"] * 1.2)
        table.append(row)
    if table:
        with (OUTPUT / "formal_metrics_comparison.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    lines = ["# P2-SFI 实验结果", "", "仅为单seed筛选；内部mAP与以下正式COCOeval AP分开记录。",
             f"本表训练轮数：{[record['training_epochs'] for record in records]}。短跑不用于判断结构收益。", "",
             "| 组别 | AP (%) | APs (%) | AP75 (%) | ΔAP (pp) | ΔAPs (pp) | GPU中位延迟(ms) | 晋级 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in table:
        lines.append(f"| {row['variant']} | {row['AP']:.3f} | {row['APs']:.3f} | {row['AP75']:.3f} | "
                     f"{row.get('delta_AP_pp', 0):+.3f} | {row.get('delta_APs_pp', 0):+.3f} | "
                     f"{row['median_ms']:.3f} | {row.get('screen_pass', False)} |")
    lines.extend(("", "晋级：ΔAP≥0.3pp、ΔAPs≥0.5pp；AP75/APm/APl各不低于基线超过0.5pp；",
                  "同口径FP32 GPU forward中位延迟增加≤20%。门槛用于工程筛选，不代表统计显著性。",
                  "训练/评估失败会停止队列；C4/C5与多seed长周期需根据结果另行决定。", ""))
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
