"""Author FP32 prediction, independent CPU COCOeval, and interleaved GPU timing."""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from common import DATA, GT, GT_SHA, RUNS, now, sha, write


def predict(name, smoke=False):
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
    weight = run / "weights/best.pt"
    class ArchiveValidator(RTDETRValidator):
        def pred_to_json(self, predn, filename):
            im_id = ids[Path(filename).stem]
            for x1, y1, x2, y2, score, cls in predn.detach().cpu().tolist():
                self.jdict.append({"image_id": im_id, "category_id": int(cls) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    model = RTDETR(str(weight))
    epoch = int(model.ckpt["epoch"]) + 1
    metric = model.val(validator=ArchiveValidator, data=str(RUNS / "smoke.yaml" if smoke else DATA),
        device="0", batch=4, imgsz=640, half=False, workers=0, save_json=True,
        plots=not smoke, project=str(run), name="evaluation", exist_ok=True, conf=.001, max_det=300)
    path = run / "evaluation/predictions.json"
    predictions = json.loads(path.read_text())
    images = {p["image_id"] for p in predictions}
    expected = 16 if smoke else 548
    assert len(images) == expected and len(predictions) == 300 * expected
    assert all(np.isfinite(v) for p in predictions for v in p["bbox"] + [p["score"]])
    write(run / "evaluation/prediction_metadata.json", {"checkpoint": str(weight),
        "checkpoint_sha256": sha(weight), "checkpoint_epoch": epoch, "gt_sha256": sha(GT),
        "predictions_sha256": sha(path), "images": expected, "detections": len(predictions),
        "native_metrics": metric.results_dict, "smoke": smoke, "finished_utc": now(),
        "protocol": "author RTDETRValidator FP32, 300 argmax queries/image, no NMS, unrounded; asserted full query coverage"})


def coco(name):
    # This subprocess imports neither torch nor the author backend.
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    run = RUNS / name
    meta = json.loads((run / "evaluation/prediction_metadata.json").read_text())
    pred = run / "evaluation/predictions.json"
    assert sha(GT) == GT_SHA == meta["gt_sha256"]
    assert sha(pred) == meta["predictions_sha256"]
    gt = COCO(str(GT))
    result = gt.loadRes(str(pred))
    images = sorted({p["image_id"] for p in result.dataset["annotations"]})
    assert len(images) == meta["images"]
    ev = COCOeval(gt, result, "bbox")
    ev.params.imgIds = images
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate(); ev.accumulate(); ev.summarize()
    names = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    per_class = {}
    for i, cat in enumerate(ev.params.catIds):
        item = {}
        for area, field in enumerate(["AP", "APs", "APm", "APl"]):
            a = ev.eval["precision"][:, :, i, area, -1]; a = a[a > -1]
            item[field] = 100 * float(a.mean()) if a.size else None
        per_class[gt.cats[cat]["name"]] = item
    write(run / "evaluation/metrics.json", {**meta, "evaluated_utc": now(), "maxDets": [1, 10, 100],
        "coco_percent": dict(zip(names, [float(v * 100) for v in ev.stats])), "per_class_percent": per_class})


def benchmark(names, output, smoke=False):
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
            "thop_gflops_unfused": get_flops(model, 640), "gate_elementwise_multiplies_per_640_image":
                0 if name.split("_")[1] == "c2" else 128 * 80 * 80, "samples_ms": []}
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
    p.add_argument("--smoke", action="store_true");a = p.parse_args()
    if a.phase == "predict":predict(a.run, a.smoke)
    elif a.phase == "coco":coco(a.run)
    else:benchmark(a.models, a.output, a.smoke)
