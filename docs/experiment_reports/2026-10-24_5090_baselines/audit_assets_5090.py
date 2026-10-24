"""Read-only 5090 baseline asset audit; image inventory is not a pixel digest.

Run with the existing UAV_CV Python. Only the explicitly selected JSON output
is written, and an existing result is never overwritten. No model is loaded.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from PIL import Image
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def audit(data_root, historical_metrics):
    errors, image_inventory, label_inventory, splits = [], [], [], {}
    label_digest = hashlib.sha256()
    data_path = data_root / "data.yaml"
    data = yaml.safe_load(data_path.read_text())
    nc = data["nc"]
    names = [data["names"][index] for index in range(nc)] if isinstance(data["names"], dict) else data["names"]
    if len(names) != nc or len(set(names)) != nc:
        raise ValueError("Dataset names must contain exactly nc unique classes")
    split_labels, image_dimensions = {}, {}
    for split in ("train", "val"):
        image_root, label_root = data_root / "images" / split, data_root / "labels" / split
        images = sorted(path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(label_root.rglob("*.txt"))
        image_keys = [path.relative_to(image_root).with_suffix("").as_posix() for path in images]
        label_keys = {path.relative_to(label_root).with_suffix("").as_posix(): path for path in labels}
        missing, extra = sorted(set(image_keys) - label_keys.keys()), sorted(label_keys.keys() - set(image_keys))
        if missing or extra or len(set(image_keys)) != len(images) or not images:
            errors.append({"split": split, "kind": "image_label_identity", "missing_labels": missing,
                           "extra_labels": extra, "duplicate_image_stems": len(images) - len(set(image_keys))})
        parsed, counts, invalid, empty, header_errors = {}, Counter(), [], 0, []
        for image in images:
            relative = image.relative_to(data_root).as_posix()
            image_inventory.append({"path": relative, "size_bytes": image.stat().st_size})
            try:
                with Image.open(image) as native:
                    width, height = native.size
                if min(width, height) < 1:
                    raise ValueError("Nonpositive image dimension")
                image_dimensions[relative] = [width, height]
            except Exception as error:
                header_errors.append({"path": relative, "error": str(error)})
        for label in labels:
            relative = label.relative_to(data_root).as_posix()
            content = label.read_bytes()
            encoded = relative.encode()
            # Length-framed relative path + exact content, independent of mtime
            # and absolute machine location; includes orphan labels too.
            label_digest.update(len(encoded).to_bytes(8, "big"))
            label_digest.update(encoded)
            label_digest.update(len(content).to_bytes(8, "big"))
            label_digest.update(content)
            label_inventory.append({"path": relative, "size_bytes": len(content)})
            rows = []
            try:
                lines = content.decode("utf-8").splitlines()
            except UnicodeError as error:
                invalid.append({"path": relative, "error": str(error)})
                lines = []
            empty += not any(line.strip() for line in lines)
            for line_id, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    row = [float(value) for value in line.split()]
                    if (len(row) != 5 or not all(math.isfinite(value) for value in row)
                            or row[0] != int(row[0]) or not 0 <= row[0] < nc
                            or not all(0 <= value <= 1 for value in row[1:])
                            or row[3] <= 0 or row[4] <= 0):
                        raise ValueError("Expected class cx cy w h; finite normalized values and positive width/height")
                    rows.append(row)
                    counts[int(row[0])] += 1
                except ValueError as error:
                    invalid.append({"path": relative, "line": line_id, "error": str(error)})
            parsed[label.relative_to(label_root).with_suffix("").as_posix()] = rows
        if invalid or header_errors:
            errors.append({"split": split, "kind": "invalid_data", "invalid_label_rows": invalid,
                           "image_header_errors": header_errors})
        splits[split] = {"images": len(images), "labels": len(labels), "boxes": sum(counts.values()),
                         "boxes_by_class_index": dict(sorted(counts.items())), "empty_label_files": empty,
                         "missing_labels": missing, "extra_labels": extra, "invalid_label_rows": len(invalid),
                         "image_headers_read": len(images) - len(header_errors), "image_header_errors": len(header_errors)}
        split_labels[split] = parsed

    annotation_path = data_root / "annotations/VisDrone2019-DET_val_coco.json"
    coco = json.loads(annotation_path.read_text())
    for field in ("images", "annotations", "categories"):
        identities = [row["id"] for row in coco[field]]
        if len(set(identities)) != len(identities):
            errors.append({"kind": "duplicate_coco_identity", "field": field})
    categories = {category["id"]: names.index(category["name"]) for category in coco["categories"]}
    if set(categories.values()) != set(range(nc)):
        errors.append({"kind": "coco_category_mapping"})
    annotations = defaultdict(list)
    valid_image_ids = {image["id"] for image in coco["images"]}
    for annotation in coco["annotations"]:
        if annotation["image_id"] not in valid_image_ids or annotation["category_id"] not in categories:
            errors.append({"kind": "coco_annotation_reference", "annotation_id": annotation["id"]})
        annotations[annotation["image_id"]].append(annotation)
    geometry_errors, coordinate_discrepancies = [], []
    max_difference, compared_boxes = 0.0, 0
    val_keys = set(split_labels["val"])
    coco_keys = [Path(image["file_name"]).with_suffix("").as_posix() for image in coco["images"]]
    if set(coco_keys) != val_keys or len(set(coco_keys)) != len(coco_keys):
        errors.append({"kind": "coco_val_identity", "missing_coco_images": sorted(val_keys - set(coco_keys)),
                       "missing_yolo_labels": sorted(set(coco_keys) - val_keys)})
    for image in coco["images"]:
        relative = "images/val/" + image["file_name"]
        if image_dimensions.get(relative) != [image["width"], image["height"]]:
            geometry_errors.append({"image_id": image["id"], "file_name": image["file_name"]})
        formal = []
        for annotation in annotations[image["id"]]:
            x, y, width, height = annotation["bbox"]
            if (not all(math.isfinite(value) for value in (x, y, width, height, annotation["area"]))
                    or width <= 0 or height <= 0 or annotation["area"] <= 0):
                errors.append({"kind": "invalid_coco_bbox_area", "annotation_id": annotation["id"]})
            formal.append([categories[annotation["category_id"]], (x + width / 2) / image["width"],
                           (y + height / 2) / image["height"], width / image["width"], height / image["height"]])
        native = split_labels["val"].get(Path(image["file_name"]).with_suffix("").as_posix(), [])
        if len(formal) != len(native):
            coordinate_discrepancies.append({"image_id": image["id"], "formal": len(formal), "yolo": len(native)})
            continue
        difference = max((abs(a - b) for first, second in zip(sorted(formal), sorted(native))
                          for a, b in zip(first, second)), default=0.0)
        max_difference = max(max_difference, difference)
        compared_boxes += len(formal)
        if difference > 5e-7 + 1e-12:
            coordinate_discrepancies.append({"image_id": image["id"], "max_normalized_field_difference": difference})
    if geometry_errors or coordinate_discrepancies:
        errors.append({"kind": "coco_yolo_disagreement", "image_dimensions": geometry_errors,
                       "coordinates": coordinate_discrepancies})
    historical = json.loads(historical_metrics.read_text())["protocol"]
    annotation_sha = file_sha(annotation_path)
    metadata = {field: {"present": sum(field in annotation for annotation in coco["annotations"]),
                        "value_counts": dict(sorted(Counter(str(annotation[field]) for annotation in coco["annotations"]
                                                          if field in annotation).items()))}
                for field in ("occlusion", "truncation")}
    return {"schema": "5090-baseline.asset-audit.v1", "status": "passed" if not errors else "failed",
            "created_utc": datetime.now(timezone.utc).isoformat(), "audit_script_sha256": file_sha(Path(__file__)),
            "data_root": str(data_root), "data_yaml": str(data_path), "data_yaml_sha256": file_sha(data_path),
            "class_names": names, "splits": splits, "label_content_sha256": label_digest.hexdigest(),
            "label_content_digest_definition": "SHA256 of sorted split/path UTF-8 and exact label bytes; each length framed by 8-byte big-endian; train then val",
            "label_inventory_sha256": json_sha(label_inventory), "image_inventory_sha256": json_sha(image_inventory),
            "image_inventory_definition": "canonical JSON list of relative filename and byte size, train then val; NOT image pixel/content SHA; headers read only",
            "coco": {"path": str(annotation_path), "sha256": annotation_sha, "images": len(coco["images"]),
                     "annotations": len(coco["annotations"]), "category_ids": sorted(categories),
                     "ignore_count": sum(bool(row.get("ignore")) for row in coco["annotations"]),
                     "iscrowd_count": sum(bool(row.get("iscrowd")) for row in coco["annotations"]),
                     "metadata_fields": metadata, "image_dimension_errors": geometry_errors,
                     "native_label_boxes_compared": compared_boxes,
                     "max_sorted_normalized_field_difference": max_difference,
                     "coordinate_tolerance": 5e-7, "floating_point_slack": 1e-12,
                     "coordinate_discrepancies": coordinate_discrepancies,
                     "pairing_definition": "per-image sorted class,cx,cy,w,h sanity check; not exact historical GT identity proof"},
            "historical_r0": {"metrics": str(historical_metrics), "annotation_sha256": historical["annotation_sha256"],
                              "annotation_sha_matches": annotation_sha == historical["annotation_sha256"],
                              "image_ids_match": sorted(valid_image_ids) == historical["image_ids"],
                              "limitation": "SHA differs; historical annotation bytes unavailable, so format-only or semantic equivalence is not established"},
            "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "SO_DETR/datasets/VisDrone2019-DET-YOLO")
    parser.add_argument("--historical-metrics", type=Path, default=ROOT / "experiments/r18_na_ablation/completed_pair/r0_formal_metrics.json")
    parser.add_argument("--output", type=Path, default=HERE / "raw/asset_audit_5090.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence: {args.output}")
    result = audit(args.data_root.resolve(), args.historical_metrics.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output), "splits": result["splits"]}, ensure_ascii=False))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
