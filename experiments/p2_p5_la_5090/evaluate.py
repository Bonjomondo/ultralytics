"""Own-checkpoint native FP32 COCO evaluation; explicit GT and validation image paths."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from experiments.decoder_budget_5090.train_5090 import now, runtime_environment, sha256_file, write_atomic
from experiments.p2_p5_la_5090.model import MODELS, inspect_model
from experiments.p2_p5_la_5090.train import NAMES, configure, source_hashes
from experiments.r18_p2_sfi.evaluate import METRICS
from experiments.r18_p2_sfi.train import tensor_digest
from ultralytics.models.rtdetr.predict import RTDETRPredictor


def category_ids(gt, names):
    by_name = {item["name"]: item["id"] for item in gt.dataset["categories"]}
    if len(by_name) != 10 or set(by_name) != set(names):
        raise ValueError("COCO categories must match the ten semantic VisDrone classes")
    return [by_name[name] for name in names]


def native_predictor(names):
    predictor = RTDETRPredictor(overrides={"imgsz": 640, "conf": .001, "max_det": 300, "verbose": False})
    # This evaluator bypasses setup_source(), which normally initializes imgsz.
    predictor.imgsz = [640, 640]
    predictor.model = SimpleNamespace(names=names)
    return predictor


def coco_metrics(gt, predictions, image_ids):
    if predictions:
        results = gt.loadRes(predictions)
    else:
        results = COCO()
        results.dataset = {"images": gt.dataset["images"], "categories": gt.dataset["categories"], "annotations": []}
        results.createIndex()
    evaluator = COCOeval(gt, results, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    per_class = {}
    for index, category in enumerate(evaluator.params.catIds):
        per_class[gt.cats[category]["name"]] = {}
        for area, label in enumerate(("AP", "APs", "APm", "APl")):
            values = evaluator.eval["precision"][:, :, index, area, -1]
            values = values[values > -1]
            per_class[gt.cats[category]["name"]][label] = float(values.mean()) if values.size else None
    return dict(zip(METRICS, map(float, evaluator.stats))), per_class


def evaluate(args):
    if args.limit < 0:
        raise ValueError("limit must be nonnegative")
    configure()
    state = json.loads((args.run / "experiment.json").read_text())
    if state["status"] != "completed" or state["variant"] != args.variant:
        raise ValueError("Require a completed training receipt for this arm")
    checkpoint = state["checkpoints"]["best"]
    weight = args.run / "weights/best.pt"
    if sha256_file(weight) != checkpoint["sha256"]:
        raise ValueError("Checkpoint hash differs from training receipt")
    args.output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(weight, map_location="cpu", weights_only=False)
    model = (saved.get("ema") or saved["model"]).float().eval().cuda()
    architecture = inspect_model(model, args.variant)
    names = [model.names[i] for i in range(10)]
    if names != NAMES:
        raise ValueError("Checkpoint class ordering differs from this experiment")
    initial = tensor_digest(model.state_dict())
    gt = COCO(str(args.annotations))
    categories = category_ids(gt, names)
    images = sorted(gt.dataset["images"], key=lambda item: item["id"])
    if len(images) != 548 or len({item["id"] for item in images}) != 548:
        raise ValueError("Require the established 548-image VisDrone validation set")
    if args.limit:
        images = images[:args.limit]
    record = {"schema": "5090.p2-p5-la.evaluation.v1", "status": "running", "variant": args.variant,
              "seed": state["seed"], "scope": "subset_gate" if args.limit else "full_val",
              "training_role": state["role"], "training_epochs": state["completed_epochs"],
              "training_fraction": state["fraction"], "architecture": architecture,
              "training_recipe": state["recipe"], "initialization": state["initialization"],
              "training_source_sha256": state["source_sha256"], "data_yaml_sha256": state["data_yaml_sha256"],
              "checkpoint": checkpoint, "training_receipt_sha256": sha256_file(args.run / "experiment.json"),
              "source_sha256": source_hashes(), "environment": runtime_environment(),
              "protocol": {"dtype": "float32", "fused": False, "batch": 2, "imgsz": 640, "conf": .001,
                           "prediction_max_det": 300, "formal_max_dets": [1, 10, 100],
                           "annotations_sha256": sha256_file(args.annotations), "images_path": str(args.images.resolve()),
                           "image_ids": [im["id"] for im in images], "category_ids": categories,
                           "preprocess": "native scale_fill RGB BCHW /255",
                           "postprocess": "native per-query argmax, sorted confidence, no NMS/no JSON rounding"}}
    write_atomic(args.output / "progress.json", record)
    predictor = native_predictor(model.names)
    predictions = []
    for start in range(0, len(images), 2):
        selected = images[start:start + 2]
        paths = [args.images / Path(im["file_name"]).name for im in selected]
        originals = [cv2.imread(str(path)) for path in paths]
        if any(im is None for im in originals):
            raise ValueError("Missing validation image in " + str(paths))
        for meta, image in zip(selected, originals):
            if image.shape[:2] != (meta["height"], meta["width"]):
                raise ValueError("Image dimensions differ from COCO GT")
        array = np.stack(predictor.pre_transform(originals))[..., ::-1].copy()
        inputs = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous().cuda().float() / 255
        output = model(inputs)[0]
        if not torch.isfinite(output).all() or output.shape != (len(selected), 300, 14):
            raise FloatingPointError("Invalid native predictions")
        predictor.batch = ([str(path) for path in paths], originals, [""] * len(paths))
        for meta, result in zip(selected, predictor.postprocess((output, None), None, originals)):
            for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                x1, y1, x2, y2 = box
                predictions.append({"image_id": meta["id"], "category_id": categories[int(cls)],
                                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    write_atomic(args.output / "predictions.json", predictions)
    metrics, per_class = coco_metrics(gt, predictions, record["protocol"]["image_ids"])
    if tensor_digest(model.state_dict()) != initial or sha256_file(weight) != checkpoint["sha256"]:
        raise RuntimeError("Evaluation mutated checkpoint or model state")
    record.update(status="completed", completed_utc=now(), metrics=metrics, per_class=per_class,
                  model_state_unchanged=True, predictions_sha256=sha256_file(args.output / "predictions.json"))
    write_atomic(args.output / "summary.json", record)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    import fcntl
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=MODELS, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    with open("/tmp/zhj-uav-5090-gpu0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with torch.inference_mode():
            evaluate(args)
