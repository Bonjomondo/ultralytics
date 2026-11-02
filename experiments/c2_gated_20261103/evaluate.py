"""Full FP32 prediction, independent COCO evaluation, and same-session timing."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from common import ARMS, REPO, REPORT, ROOT, RUNS, now, sha, write

GT = REPO.parent / "datasets/VisDrone2019-DET-YOLO/annotations/VisDrone2019-DET_val_coco.json"
GT_SHA = "6f57aa4b42d64ee7b99b8a6a5b159ad19cba2eacf00bd86f23314966614bf4d8"


def predict(arm):
    from ultralytics import RTDETR
    from experiments.r18_p2_sfi.evaluate import ConfidenceAuditValidator
    assert sha(GT) == GT_SHA
    run = RUNS / arm
    weight = run / "weights/best.pt"
    ckpt = torch.load(weight, map_location="cpu")
    epoch = int(ckpt["epoch"]) + 1
    del ckpt
    wrapper = RTDETR(str(weight))
    audit = []
    wrapper.add_callback("on_val_end", lambda validator: audit.extend(validator.filter_audit))
    result = wrapper.val(validator=ConfidenceAuditValidator, data=str(ROOT / "configs/visdrone-local.yaml"),
                         device="0", imgsz=640, batch=2, workers=0, half=False, conf=0.0,
                         max_det=300, plots=True, save_json=True, verbose=False,
                         project=str(run), name="formal_native", exist_ok=True)
    source = Path(result.save_dir) / "predictions.json"
    predictions = json.loads(source.read_text())
    gt = json.loads(GT.read_text())
    ids = {Path(x["file_name"]).name: x["id"] for x in gt["images"]}
    assert len(ids) == 548 and len(audit) == 548
    assert len(predictions) == 548 * 300
    assert len({a["file_name"] for a in audit}) == 548
    for item in predictions:
        item["image_id"] = ids[item["file_name"]]
    mapped = run / "evaluation/predictions_coco.json"
    write(mapped, predictions)
    write(run / "evaluation/prediction_receipt.json", {
        "arm": arm, "checkpoint": str(weight), "checkpoint_sha256": sha(weight), "best_epoch": epoch,
        "gt_sha256": sha(GT), "prediction_sha256": sha(mapped), "images": 548,
        "detections": len(predictions), "native_metrics": result.results_dict,
        "protocol": "FP32 fused, conf0, 300 argmax queries/image, no NMS, converted VisDrone val548",
    })


def coco(arm):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    assert sha(GT) == GT_SHA
    run = RUNS / arm
    receipt = json.loads((run / "evaluation/prediction_receipt.json").read_text())
    assert sha(run / "weights/best.pt") == receipt["checkpoint_sha256"]
    assert sha(run / "evaluation/predictions_coco.json") == receipt["prediction_sha256"]
    gt = COCO(str(GT))
    ev = COCOeval(gt, gt.loadRes(str(run / "evaluation/predictions_coco.json")), "bbox")
    ev.params.imgIds = sorted(gt.getImgIds())
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    names = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    per_class = {}
    for i, cat in enumerate(ev.params.catIds):
        per_class[gt.cats[cat]["name"]] = {}
        for area, name in enumerate(["AP", "APs", "APm", "APl"]):
            a = ev.eval["precision"][:, :, i, area, -1]
            a = a[a > -1]
            per_class[gt.cats[cat]["name"]][name] = 100 * float(a.mean()) if a.size else None
    write(REPORT / f"raw/{arm}/formal_metrics.json", {
        **receipt, "evaluated_utc": now(), "maxDets": [1, 10, 100],
        "coco_percent": dict(zip(names, (100 * float(v) for v in ev.stats))),
        "per_class_percent": per_class,
    })


def benchmark(smoke=False):
    from ultralytics import RTDETR
    from ultralytics.utils.torch_utils import get_flops
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    models, records = {}, {}
    x = torch.rand(1, 3, 640, 640, device="cuda")
    for arm in ARMS:
        weight = RUNS / ("smoke_" + arm if smoke else arm) / "weights/best.pt"
        m = RTDETR(str(weight)).model.float().cuda().eval()
        records[arm] = {"parameters": sum(p.numel() for p in m.parameters()),
                        "gflops": get_flops(m, imgsz=640), "checkpoint_sha256": sha(weight), "samples_ms": []}
        m.fuse(verbose=False)
        models[arm] = m
    with torch.inference_mode():
        for m in models.values():
            for _ in range(50):
                m(x)
        torch.cuda.synchronize()
        rng = random.Random(11)
        orders = []
        for _ in range(5):
            order = list(ARMS)
            rng.shuffle(order)
            orders.append(order)
            for arm in order:
                m = models[arm]
                for _ in range(10):
                    m(x)
                for _ in range(100):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    m(x)
                    end.record()
                    end.synchronize()
                    records[arm]["samples_ms"].append(float(start.elapsed_time(end)))
    for r in records.values():
        v = r["samples_ms"]
        r.update(mean_ms=float(np.mean(v)), median_ms=float(np.median(v)), p95_ms=float(np.percentile(v, 95)))
    write(REPORT / ("raw/smoke_benchmark.json" if smoke else "raw/benchmark.json"), {
        "models": records, "orders": orders, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "smoke_only_no_accuracy_conclusion": smoke,
        "protocol": "FP32 fused batch1 640x640 resident input; CUDA events; excludes transfer/pre/postprocess; 5x100/model",
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["predict", "coco", "benchmark"])
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--smoke", action="store_true", help="Time the four short-run checkpoints, for pipeline verification only")
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.phase == "benchmark":
        benchmark(args.smoke)
    else:
        {"predict": predict, "coco": coco}[args.phase](args.arm)
