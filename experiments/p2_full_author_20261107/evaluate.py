"""Author FP32 prediction, independent CPU COCOeval, and interleaved GPU timing."""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from common import DATA, GT, GT_SHA, RUNS, now, sha, write


def predict(name, budget=350, kind="best", smoke=False):
    import models
    import torch
    from ultralytics import RTDETR
    from ultralytics.models.rtdetr.val import RTDETRValidator
    from ultralytics.utils import callbacks
    callbacks.add_integration_callbacks = lambda instance: None
    torch.set_num_threads(8)
    assert sha(GT) == GT_SHA
    gt = json.loads(GT.read_text())
    ids = {Path(x["file_name"]).stem: x["id"] for x in gt["images"]}
    run = RUNS / name
    weight = run / ("weights/best.pt" if smoke else f"milestones/e{budget}/{kind}.pt")
    output = run / "evaluation" / f"e{budget}_{kind}"
    class ArchiveValidator(RTDETRValidator):
        def postprocess(self, preds):
            outputs = super().postprocess(preds)
            scores = preds[0][..., 4:].max(-1).values
            self.sorted_levels = []
            for i, levels in enumerate(head.last_query_levels):
                order = scores[i].argsort(descending=True).cpu().tolist()
                self.sorted_levels.append([head.feature_strides[levels[j]] for j in order])
            self.query_image_index = 0
            assert len(self.sorted_levels) == len(outputs)
            return outputs

        def pred_to_json(self, predn, filename):
            im_id = ids[Path(filename).stem]
            for j, (x1, y1, x2, y2, score, cls) in enumerate(predn.detach().cpu().tolist()):
                self.jdict.append({"image_id": im_id, "category_id": int(cls) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score,
                    "query_stride": self.sorted_levels[self.query_image_index][j]})
            self.query_image_index += 1
    model = RTDETR(str(weight))
    epoch = int(model.ckpt["epoch"]) + 1
    head = model.model.model[-1]
    head.capture_query_stats = True
    metric = model.val(validator=ArchiveValidator, data=str(RUNS / "smoke.yaml" if smoke else DATA),
        device="0", batch=4, imgsz=640, half=False, workers=0, save_json=True,
        plots=(not smoke and budget == 350 and kind == "best"), project=str(run), name=f"evaluation/e{budget}_{kind}", exist_ok=True, conf=.001, max_det=300)
    path = output / "predictions.json"
    predictions = json.loads(path.read_text())
    images = {p["image_id"] for p in predictions}
    expected = 16 if smoke else 548
    assert len(images) == expected and len(predictions) == 300 * expected
    assert all(np.isfinite(v) for p in predictions for v in p["bbox"] + [p["score"]])
    write(output / "prediction_metadata.json", {"checkpoint": str(weight), "budget":budget, "kind":kind,
        "checkpoint_sha256": sha(weight), "checkpoint_epoch": epoch, "gt_sha256": sha(GT),
        "predictions_sha256": sha(path), "images": expected, "detections": len(predictions),
        "native_metrics": metric.results_dict, "smoke": smoke, "finished_utc": now(),
        "protocol": "author RTDETRValidator FP32, 300 argmax queries/image, no NMS, unrounded; asserted full query coverage"})


def tiny_recall(ev, gt, result):
    """Recall of resized GT by short side, using COCO's class-wise one-to-one matching."""
    bins = [(0, 8), (8, 16), (16, 32), (32, float("inf"))]
    rows = [{"short_side_640":f"[{lo},{hi})", "gt":0, "matched_iou50":0,
             "matched_iou75":0, "matched_query_strides_iou50":{}} for lo,hi in bins]
    for item in ev.evalImgs:
        if item is None or list(item["aRng"]) != [0, 1e10]:
            continue
        image = gt.imgs[item["image_id"]]
        for j, gt_id in enumerate(item["gtIds"]):
            if item["gtIgnore"][j]:
                continue
            box = gt.anns[gt_id]["bbox"]
            side = min(box[2] * 640 / image["width"], box[3] * 640 / image["height"])
            index = next(i for i,(lo,hi) in enumerate(bins) if lo <= side < hi)
            row = rows[index]
            row["gt"] += 1
            for threshold, field in [(0, "matched_iou50"), (5, "matched_iou75")]:
                row[field] += int(item["gtMatches"][threshold, j] > 0)
            detection = int(item["gtMatches"][0, j])
            if detection:
                stride = str(result.anns[detection]["query_stride"])
                row["matched_query_strides_iou50"][stride] = row["matched_query_strides_iou50"].get(stride, 0) + 1
    for row in rows:
        for suffix in ["50", "75"]:
            row[f"recall_iou{suffix}_percent"] = 100 * row[f"matched_iou{suffix}"] / row["gt"] if row["gt"] else None
    return rows


def coco(name, budget=350, kind="best"):
    # This process intentionally imports neither torch nor the author engine.
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    output = RUNS / name / "evaluation" / f"e{budget}_{kind}"
    meta = json.loads((output / "prediction_metadata.json").read_text())
    pred = output / "predictions.json"
    assert sha(GT) == GT_SHA == meta["gt_sha256"]
    assert sha(pred) == meta["predictions_sha256"]
    gt = COCO(str(GT))
    result = gt.loadRes(str(pred))
    images = sorted({p["image_id"] for p in result.dataset["annotations"]})
    assert len(images) == meta["images"]
    names = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    records = {}
    per_class = {}
    for maximum in [100, 300]:
        ev = COCOeval(gt, result, "bbox")
        ev.params.imgIds = images
        ev.params.maxDets = [1, 10, maximum]
        ev.evaluate(); ev.accumulate(); ev.summarize()
        # COCO summarize hardcodes maxDets100 for stats[0]; calculate AP explicitly for maxDets300.
        precision = ev.eval["precision"][:, :, :, 0, -1]
        valid = precision[precision > -1]
        stats = [float(v*100) for v in ev.stats]
        stats[0] = float(valid.mean()*100) if valid.size else -1.
        metric_names = ["AR300" if name == "AR100" and maximum == 300 else name for name in names]
        records[str(maximum)] = {"metrics":dict(zip(metric_names, stats)), "tiny_recall":tiny_recall(ev, gt, result)}
        if maximum == 100:
            for i, cat in enumerate(ev.params.catIds):
                values = {}
                for area, field in enumerate(["AP", "APs", "APm", "APl"]):
                    a = ev.eval["precision"][:, :, i, area, -1]; a = a[a > -1]
                    values[field] = float(a.mean()*100) if a.size else None
                per_class[gt.cats[cat]["name"]] = values
    write(output / "metrics.json", {**meta, "evaluated_utc":now(),
          "maxDets":[1,10,100], "coco_percent":records["100"]["metrics"],
          "supplement_maxdet300":records["300"], "tiny_recall_maxdet100":records["100"]["tiny_recall"],
          "per_class_percent":per_class,
          "tiny_protocol":"640 square resize, class-wise COCO matching, ignored GT excluded; query stride denotes initialization origin, not exclusive causal attribution"})


def benchmark(names, output, smoke=False):
    import models as model_definitions
    import torch
    from ultralytics import RTDETR
    from ultralytics.utils.torch_utils import get_flops
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    x = torch.rand(1, 3, 640, 640, device="cuda")
    models, records = {}, {}
    for name in names:
        weight = RUNS / name / "weights/best.pt"
        model = RTDETR(str(weight)).model.float().cuda().eval()
        records[name] = {"checkpoint_sha256": sha(weight), "parameters": sum(p.numel() for p in model.parameters()),
            "thop_gflops_unfused": get_flops(model, 640), "feature_strides": model.model[-1].feature_strides, "samples_ms": []}
        models[name] = model.fuse().eval()
    orders = []; rng = random.Random(20261105)
    with torch.inference_mode():
        for model in models.values():
            for _ in range(2 if smoke else 50):model(x)
        for _ in range(1 if smoke else 5):
            order = list(names); rng.shuffle(order); orders.append(order)
            for name in order:
                for _ in range(2 if smoke else 10):models[name](x)
                for _ in range(8 if smoke else 100):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record(); models[name](x); e.record(); e.synchronize()
                    records[name]["samples_ms"].append(float(s.elapsed_time(e)))
    for row in records.values():
        values = row["samples_ms"]
        row.update(mean_ms=float(np.mean(values)), median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)))
    write(RUNS / output, {"models": records, "orders": orders, "smoke": smoke, "finished_utc": now(),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "protocol": "batch1 640 fused FP32 resident GPU input, CUDA event synchronized, randomized interleaved rounds; excludes pre/postprocess and transfers"})


if __name__ == "__main__":
    p = argparse.ArgumentParser();p.add_argument("phase", choices=["predict", "coco", "benchmark"])
    p.add_argument("--run");p.add_argument("--models", nargs="+");p.add_argument("--output", default="benchmark_screen.json")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--budget", type=int, default=350)
    p.add_argument("--kind", choices=["best", "last"], default="best");a = p.parse_args()
    if a.phase == "predict":predict(a.run, a.budget, a.kind, a.smoke)
    elif a.phase == "coco":coco(a.run, a.budget, a.kind)
    else:benchmark(a.models, a.output, a.smoke)
