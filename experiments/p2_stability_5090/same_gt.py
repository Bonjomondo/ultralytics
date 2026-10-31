"""Offline same-GT comparison using COCO's score-ordered, per-category matching."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "SO_DETR/datasets/VisDrone2019-DET-YOLO"
OLD = ROOT / "runs/decoder_budget_5090/5090-c0c2-baselines-20261024-s0-r1"
JOINT = ROOT / "runs/p2_p5_la_5090/5090-c2-la-150e-20261026-s0-r1"
LA = ROOT / "runs/p2_p5_la_5090/5090-la-only-150e-20261027-s0-r1"
DEPTH = ROOT / "runs/diagnostics/5090-depth-eval-c2-20261025-final"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def table(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def defaults():
    models = {}
    for arm in ("c0", "c2"):
        run = OLD / f"5090-rtdetr-r18-{arm}-150e-s0-query-full"
        record = read(run / "summary.json")
        condition = next(x for x in record["conditions"] if x["queries"] == 300)
        models[arm] = {"path": str(run / "q300/predictions.json"),
                       "sha256": condition["predictions_file_sha256"],
                       "annotation_sha256": record["protocol"]["annotation_sha256"]}
    for arm, directory in (("la", LA), ("c2-la", JOINT)):
        record = read(directory / "evaluation/summary.json")
        models[arm] = {"path": str(directory / "evaluation/predictions.json"),
                       "sha256": record["predictions_sha256"],
                       "annotation_sha256": record["protocol"]["annotations_sha256"]}
    for arm in ("c0", "c2"):
        directory = ROOT / f"runs/diagnostics/5090-depth-eval-{arm}-20261025-final"
        record = read(directory / "summary.json")
        for depth in ("d1", "d2"):
            models[arm + "-" + depth] = {"path": str(directory / (depth + "-predictions.json")),
                    "sha256": record["conditions"][depth]["predictions"]["sha256"],
                    "annotation_sha256": record["formal_protocol"]["annotation_sha256"]}
    return models


def match(gt, predictions):
    """Match once at 300/category; earlier score prefixes have identical matches."""
    if not predictions:
        return {(.5, 100): {}, (.75, 100): {}, (.5, 300): {}, (.75, 300): {}}
    known_images, known_categories = set(gt.imgs), set(gt.cats)
    for p in predictions:
        if (p["image_id"] not in known_images or p["category_id"] not in known_categories
                or len(p["bbox"]) != 4 or not all(math.isfinite(v) for v in [*p["bbox"], p["score"]])
                or min(p["bbox"][2:]) <= 0 or not 0 <= p["score"] <= 1):
            raise ValueError("Invalid prediction identity, geometry or score")
    detections = gt.loadRes(predictions)
    evaluator = COCOeval(gt, detections, "bbox")
    evaluator.params.imgIds = sorted(gt.imgs)
    evaluator.params.iouThrs = np.array([.5, .75])
    evaluator.params.areaRng = [[0, 1e10]]
    evaluator.params.areaRngLbl = ["all"]
    evaluator.params.maxDets = [1, 10, 300]
    evaluator.evaluate()
    states = {(iou, budget): {} for iou in (.5, .75) for budget in (100, 300)}
    for image in evaluator.evalImgs:
        if image is None:
            continue
        gt_index = {int(g): i for i, g in enumerate(image["gtIds"])}
        for ti, iou in enumerate((.5, .75)):
            for rank, did in enumerate(image["dtIds"], 1):
                gid = int(image["dtMatches"][ti, rank - 1])
                if not gid or image["gtIgnore"][gt_index[gid]] or image["dtIgnore"][ti, rank - 1]:
                    continue
                pred, target = detections.anns[did], gt.anns[gid]
                overlap = float(mask_utils.iou([pred["bbox"]], [target["bbox"]], [0])[0, 0])
                for budget in (100, 300):
                    if rank <= budget:
                        states[iou, budget][gid] = {"score": pred["score"], "iou": overlap}
    return states


def attributes(gt):
    counts = Counter(a["image_id"] for a in gt.anns.values() if not a.get("iscrowd") and not a.get("ignore"))
    result = {}
    for gid, a in gt.anns.items():
        if a.get("iscrowd") or a.get("ignore"):
            continue
        im = gt.imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        side = math.sqrt(a["area"])
        edge = min(x, y, im["width"] - x - w, im["height"] - y - h)
        n = counts[a["image_id"]]
        result[gid] = {"all": "all", "class": gt.cats[a["category_id"]]["name"],
                       "coco_size": "small" if side < 32 else "medium" if side < 96 else "large",
                       "sqrt_area_px": "<8" if side < 8 else "8-16" if side < 16 else "16-32" if side < 32 else ">=32",
                       "image_gt_count": "<50" if n < 50 else "50-100" if n < 100 else ">=100",
                       "edge_px": "<16" if edge < 16 else "16-32" if edge < 32 else ">=32",
                       **{key: str(a.get(key, "missing")) for key in ("occlusion", "truncation")}}
    return result


def compare(left, right, groups, conf):
    counts = defaultdict(Counter)
    for gid, attrs in groups.items():
        a, b = left.get(gid), right.get(gid)
        a = a if a and a["score"] >= conf else None
        b = b if b and b["score"] >= conf else None
        status = "both" if a and b else "lost" if a else "recovered" if b else "neither"
        for dimension, group in attrs.items():
            c = counts[dimension, group]
            c["gt_count"] += 1
            c[status] += 1
            if a and b:
                c["score_delta_sum"] += b["score"] - a["score"]
                c["iou_delta_sum"] += b["iou"] - a["iou"]
    rows = []
    for (dimension, group), c in sorted(counts.items()):
        rows.append({"dimension": dimension, "group": group, "gt_count": c["gt_count"],
                     **{key: c[key] for key in ("recovered", "lost", "both", "neither")},
                     "net_recovered": c["recovered"] - c["lost"],
                     "recall_delta_pp": 100 * (c["recovered"] - c["lost"]) / c["gt_count"],
                     "both_score_delta_mean": c["score_delta_sum"] / c["both"] if c["both"] else None,
                     "both_iou_delta_mean": c["iou_delta_sum"] / c["both"] if c["both"] else None})
    return rows


def run(args):
    models = read(args.inputs) if args.inputs else defaults()
    gt_sha = sha(args.annotations)
    for name, artifact in models.items():
        if artifact["annotation_sha256"] != gt_sha or sha(artifact["path"]) != artifact["sha256"]:
            raise ValueError("Prediction/GT receipt mismatch: " + name)
    args.output.mkdir(parents=True, exist_ok=False)
    gt = COCO(str(args.annotations))
    groups = attributes(gt)
    states = {name: match(gt, read(artifact["path"])) for name, artifact in models.items()}
    pairs = (("c0", "c2"), ("la", "c2-la"), ("c2", "c2-la"),
             ("c0-d1", "c2-d1"), ("c0-d2", "c2-d2"),
             ("c0-d1", "c0-d2"), ("c0-d2", "c0"), ("c2-d1", "c2-d2"), ("c2-d2", "c2"))
    rows = []
    for a, b in pairs:
        for iou in (.5, .75):
            for budget in (100, 300):
                for conf in (.001, .05, .25):
                    rows.extend({"pair": a + "->" + b, "iou": iou, "max_dets_per_category": budget,
                                 "conf": conf, **r} for r in compare(states[a][iou, budget], states[b][iou, budget], groups, conf))
    table(args.output / "groups.csv", rows)
    main = [r for r in rows if r["dimension"] == "all" and r["conf"] == .001 and r["max_dets_per_category"] == 100]
    table(args.output / "overall.csv", main)
    coverage, stages = [], []
    for arm in ("c0", "c2"):
        receipt = read(OLD / f"5090-rtdetr-r18-{arm}-150e-s0-query-full/summary.json")
        condition = next(x for x in receipt["conditions"] if x["queries"] == 300)
        coverage.extend({"model": arm, **r} for r in condition["coverage_all_gt"])
        directory = ROOT / f"runs/diagnostics/5090-depth-eval-{arm}-20261025-final"
        for depth in ("d1", "d2", "d3"):
            metrics = read(directory / (depth + "-coco.json"))["metrics"]
            stages.append({"model": arm, "depth": depth, **{k: 100 * v for k, v in metrics.items()}})
    table(args.output / "encoder_topk_coverage.csv", coverage)
    table(args.output / "native_decoder_stages.csv", stages)
    # Intersection uses stable GT IDs, never cross-model query IDs.
    intersections = []
    for iou in (.5, .75):
        s = {name: set(value[iou, 100]) for name, value in states.items()}
        a, b = s["c2"] - s["c0"], s["c2-la"] - s["la"]
        intersections.append({"iou": iou, "p2_aifi_recovered": len(a), "p2_la_recovered": len(b),
                              "recovered_in_both": len(a & b), "union": len(a | b)})
    write(args.output / "summary.json", {"status": "completed", "images": len(gt.imgs), "valid_gt": len(groups),
          "annotation_sha256": gt_sha, "inputs": models, "same_gt_recovery_overlap": intersections,
          "protocol": {"matching": "COCO greedy score-order matching within image and category; iscrowd/ignored GT excluded",
                       "budgets": [100, 300], "confidence": [.001, .05, .25], "iou": [.5, .75],
                       "sizes": "original image coordinates; sqrt(area) bins; COCO group cutoffs 32 and 96",
                       "density": "valid annotated GT count per image, not local neighbor density",
                       "limitation": "Recall/state diagnostic, not AP or causal mechanism evidence. Budget300 is diagnostic only."}})
    lines = ["# 同 GT 离线诊断", "", f"完整{len(gt.imgs)}图，{len(groups)}个有效GT。conf≥0.001，每图每类别最多100框，COCO匹配。", "",
             "| 对比 | IoU | 找回 | 丢失 | 净找回 | 召回变化pp |", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['pair']} | {r['iou']} | {r['recovered']} | {r['lost']} | {r['net_recovered']} | {r['recall_delta_pp']:.3f} |" for r in main]
    lines += ["", "groups.csv包含类别、原图尺寸、图像GT数量、边缘距离、遮挡、截断，以及100/300检测预算和三档置信度敏感性。",
              "", "这是固定阈值下的GT状态/召回诊断，不是AP变化；300预算只用于诊断。共有检出框的IoU/分数差使用各模型独立匹配，不能当作固定Query轨迹。"]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, default=DATA / "annotations/VisDrone2019-DET_val_coco.json")
    p.add_argument("--inputs", type=Path)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
