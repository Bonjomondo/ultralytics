#!/usr/bin/env python3
"""Formal COCOeval for the paper-replication RT-DETR baseline on VisDrone.

Runs *after* training finishes (it evaluates a saved `.pt` on the val split with
pycocotools-style COCO metrics: AP, AP50, AP75, APs, APm, APl). It is a separate
post-hoc script so it never touches the training run.

Steps:
  1. Build a COCO ground-truth JSON from the VisDrone val YOLO labels.
  2. Run the model over the val images and collect COCO-format detections.
  3. Evaluate with `faster_coco_eval` (COCO `iouType='bbox'`).

Example
-------
::

    PYTHONPATH=$PWD python eval_coco.py \
        --weights runs/detect/paper_repro/rtdetr-r50-paper-350e-s0/weights/best.pt \
        --data configs/visdrone-local.yaml --imgsz 640 --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

from faster_coco_eval import COCO, COCOeval_faster
from ultralytics import RTDETR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="Path to best.pt / last.pt.")
    p.add_argument("--data", required=True, help="Dataset YAML (VisDrone).")
    p.add_argument("--split", default="val", help="Split to evaluate: val or test.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--conf", type=float, default=0.001, help="Low conf for COCO recall (COCO protocol).")
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--out", default=None, help="Output dir for gt/pred JSON (default: next to weights).")
    p.add_argument("--limit", type=int, default=0, help="Debug: use only first N images (0 = all).")
    return p.parse_args()


def build_gt(data_cfg: dict, split: str, out_dir: Path, limit: int = 0) -> tuple[Path, list[dict]]:
    """Convert VisDrone YOLO labels for `split` into a COCO ground-truth JSON."""
    root = Path(data_cfg["path"])
    img_dir = root / data_cfg[split]
    lbl_dir = img_dir.parent / "labels"

    names = data_cfg["names"]
    if isinstance(names, dict):  # yaml maps {0: pedestrian, ...}
        names = [names[i] for i in sorted(names)]
    categories = [{"id": i + 1, "name": str(n)} for i, n in enumerate(names)]

    images, annotations = [], []
    ann_id = 1
    files = sorted(img_dir.glob("*.jpg"))
    if limit:
        files = files[:limit]

    for idx, img_path in enumerate(tqdm(files, desc="gt")):
        with Image.open(img_path) as im:
            w, h = im.size
        image_id = idx + 1
        images.append({"id": image_id, "file_name": img_path.name, "width": w, "height": h})

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            x = (cx - bw / 2) * w
            y = (cy - bh / 2) * h
            bw, bh = bw * w, bh * h
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": c + 1,
                    "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    gt = {"images": images, "annotations": annotations, "categories": categories}
    gt_path = out_dir / f"gt_{split}.json"
    gt_path.write_text(json.dumps(gt))
    return gt_path, images


def build_predictions(args, images: list[dict], out_dir: Path, split: str) -> Path:
    """Run the model over the val images and dump COCO-format detections."""
    root = Path(yaml.safe_load(Path(args.data).read_text())["path"])
    img_dir = root / yaml.safe_load(Path(args.data).read_text())[split]

    stem_to_id = {Path(im["file_name"]).stem: im["id"] for im in images}
    model = RTDETR(args.weights)

    preds = []
    results = model.predict(
        source=str(img_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        batch=args.batch,
        stream=True,
        verbose=False,
    )
    for r in tqdm(results, desc="pred", total=len(stem_to_id)):
        stem = Path(r.path).stem
        image_id = stem_to_id.get(stem)
        if image_id is None:
            continue
        boxes = r.boxes
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, conf):
            preds.append(
                {
                    "image_id": image_id,
                    "category_id": int(c) + 1,
                    "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                    "score": float(s),
                }
            )

    pred_path = out_dir / f"pred_{split}.json"
    pred_path.write_text(json.dumps(preds))
    return pred_path


def main() -> None:
    args = parse_args()
    data_cfg = yaml.safe_load(Path(args.data).read_text())

    out_dir = Path(args.out) if args.out else Path(args.weights).parent.parent / "coco_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path, images = build_gt(data_cfg, args.split, out_dir, args.limit)
    pred_path = build_predictions(args, images, out_dir, args.split)
    print(f"GT   : {gt_path}\nPreds: {pred_path}")

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(pred_path))
    ev = COCOeval_faster(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    stats = ev.stats
    labels = ["AP@[.5:.95]", "AP50", "AP75", "APs", "APm", "APl"]
    print("\n=== Formal COCO metrics ===")
    for name, val in zip(labels, stats[:6]):
        print(f"  {name:12s}: {val * 100:.2f}")


if __name__ == "__main__":
    main()