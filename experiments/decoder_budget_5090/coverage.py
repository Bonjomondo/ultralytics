"""Offline candidate coverage for saved, score-ordered encoder proposals.

Coverage is many-to-many geometry, not COCO matching or an AP estimate. A
proposal may cover several neighbouring GTs. Prefixes describe the saved Top-K
selection only; tied scores can make a separate native Top-K(K) choose different
tokens. Query-budget experiments must still execute the complete decoder.
"""
import math

import torch


def _tensor(value, *, name, shape=None, dtype=torch.float64):
    result = torch.as_tensor(value).detach().cpu().to(dtype)
    if shape is not None and tuple(result.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}")
    if result.is_floating_point() and not torch.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _boxes(value, name):
    result = _tensor(value, name=name)
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError(f"{name} must be normalized cxcywh [N,4]")
    if (result[:, 2:] <= 0).any():
        raise ValueError(f"{name} widths and heights must be positive")
    return result


def _iou(proposals, targets):
    first = torch.cat((proposals[:, :2] - proposals[:, 2:] / 2,
                       proposals[:, :2] + proposals[:, 2:] / 2), -1)
    second = torch.cat((targets[:, :2] - targets[:, 2:] / 2,
                        targets[:, :2] + targets[:, 2:] / 2), -1)
    intersection = (torch.minimum(first[:, None, 2:], second[None, :, 2:]) -
                    torch.maximum(first[:, None, :2], second[None, :, :2])).clamp(min=0).prod(-1)
    union = proposals[:, 2:].prod(-1)[:, None] + targets[:, 2:].prod(-1)[None, :] - intersection
    return intersection / union.clamp(min=torch.finfo(torch.float64).tiny)


def _flags(targets, key, count):
    value = torch.as_tensor(targets.get(key, torch.zeros(count, dtype=torch.bool))).detach().cpu()
    if value.shape != (count,) or not ((value == 0) | (value == 1)).all():
        raise ValueError(f"targets {key} must contain boolean [G] flags")
    return value.bool()


def topk_coverage(boxes, logits, targets, image_size, *, budgets=(150, 200, 250, 300),
                  thresholds=(0.3, 0.5, 0.75), density_radius=0.05):
    """Measure saved encoder prefixes against original-image GT identities.

    ``boxes`` and ``targets['boxes']`` are normalized cxcywh; ``logits`` are
    sigmoid-classification logits in the original native selection order. The
    caller must provide ``targets['gt_ids']`` and integer model-column classes.
    Supplied original-pixel ``areas`` are preferred to normalized box areas.
    Ignore/iscrowd GTs are excluded from every denominator and density grouping.

    Rows are additive across images by summing counts/denominators, never by
    averaging per-image fractions. ``None`` denotes a zero denominator. New
    coverage is relative to the preceding requested budget (zero initially).
    """
    boxes = _boxes(boxes, "boxes")
    logits = _tensor(logits, name="logits")
    if logits.ndim != 2 or logits.shape[0] != len(boxes) or logits.shape[1] < 1:
        raise ValueError("logits must have shape [Q,C] with C >= 1")
    ranking_scores = logits.max(-1).values
    if (ranking_scores[1:] > ranking_scores[:-1]).any():
        raise ValueError("Encoder proposals must retain descending native max-logit order")
    budgets, thresholds = list(budgets), list(thresholds)
    if (not budgets or any(type(k) is not int or not 0 <= k <= len(boxes) for k in budgets)
            or budgets != sorted(set(budgets))):
        raise ValueError("budgets must be unique increasing integer prefixes in [0,Q]")
    if (not thresholds or any(isinstance(t, bool) or not isinstance(t, (int, float))
                              or not math.isfinite(t) or not 0 < t <= 1 for t in thresholds)
            or thresholds != sorted(set(thresholds))):
        raise ValueError("thresholds must be unique increasing numbers in (0,1]")
    if (len(image_size) != 2 or any(type(s) is not int or s <= 0 for s in image_size)):
        raise ValueError("image_size must contain positive original integer height/width")
    if (isinstance(density_radius, bool) or not isinstance(density_radius, (int, float))
            or not math.isfinite(density_radius) or density_radius <= 0):
        raise ValueError("density_radius must be a positive finite image-diagonal fraction")
    gt_boxes = _boxes(targets["boxes"], "target boxes")
    count = len(gt_boxes)
    classes_raw = torch.as_tensor(targets["classes"])
    if classes_raw.shape != (count,) or classes_raw.dtype not in (torch.int32, torch.int64):
        raise ValueError("target classes must be integer model columns [G]")
    classes = classes_raw.detach().cpu().long()
    if ((classes < 0) | (classes >= logits.shape[1])).any():
        raise ValueError("target classes must identify a model logit column")
    gt_ids = list(targets["gt_ids"])
    if (len(gt_ids) != count or any(isinstance(i, bool) or not isinstance(i, (int, str)) for i in gt_ids)
            or len(set(map(str, gt_ids))) != count):
        raise ValueError("gt_ids must contain unique integer/string identities [G]")
    eligible = ~(_flags(targets, "ignore", count) | _flags(targets, "iscrowd", count))
    height, width = image_size
    areas = _tensor(targets.get("areas", gt_boxes[:, 2] * width * gt_boxes[:, 3] * height),
                    name="target areas", shape=(count,))
    if (areas < 0).any():
        raise ValueError("target areas must be nonnegative original-pixel areas")
    groups = {"all": eligible}
    for cls in range(logits.shape[1]):
        groups[f"class:{cls}"] = eligible & (classes == cls)
    groups.update({"area:small": eligible & (areas < 32 ** 2),
                   "area:medium": eligible & (areas >= 32 ** 2) & (areas < 96 ** 2),
                   "area:large": eligible & (areas >= 96 ** 2)})
    # Neighbours are other non-ignored, non-crowd GTs, regardless of class.
    centers = gt_boxes[:, :2] * gt_boxes.new_tensor([width, height])
    distances = torch.cdist(centers, centers)
    if count:
        distances.fill_diagonal_(float("inf"))
        distances[:, ~eligible] = float("inf")
    nearby = (distances <= density_radius * math.hypot(width, height)).any(-1)
    groups["density:nearby"] = eligible & nearby
    groups["density:isolated"] = eligible & ~nearby
    ious = _iou(boxes, gt_boxes)
    class_agrees = logits.argmax(-1)[:, None] == classes[None, :]
    rows, per_gt = [], []
    for threshold in thresholds:
        matches = ious >= threshold
        class_matches = matches & class_agrees
        previous = torch.zeros(count, dtype=torch.bool)
        previous_correct = previous.clone()
        first_rank = torch.full((count,), -1, dtype=torch.int64)
        first_correct_rank = first_rank.clone()
        for rank in range(len(boxes)):
            first_rank[(first_rank < 0) & matches[rank]] = rank + 1
            first_correct_rank[(first_correct_rank < 0) & class_matches[rank]] = rank + 1
        for budget in budgets:
            covered, correct = matches[:budget].any(0), class_matches[:budget].any(0)
            for group, mask in groups.items():
                denominator = int(mask.sum())
                ncovered, ncorrect = int((mask & covered).sum()), int((mask & correct).sum())
                rows.append({"budget": budget, "iou_threshold": float(threshold), "group": group,
                             "denominator": denominator, "covered": ncovered,
                             "class_correct_covered": ncorrect,
                             "coverage": ncovered / denominator if denominator else None,
                             "class_correct_coverage": ncorrect / denominator if denominator else None,
                             "newly_covered": int((mask & covered & ~previous).sum()),
                             "newly_class_correct_covered": int((mask & correct & ~previous_correct).sum())})
            previous, previous_correct = covered, correct
        for index, identity in enumerate(gt_ids):
            per_gt.append({"gt_id": identity, "class_index": int(classes[index]),
                           "eligible": bool(eligible[index]), "iou_threshold": float(threshold),
                           "first_covered_rank": int(first_rank[index]) if first_rank[index] >= 0 else None,
                           "first_class_correct_rank": int(first_correct_rank[index]) if first_correct_rank[index] >= 0 else None})
    return {"schema": "decoder-budget.topk-coverage.v1", "candidate_count": len(boxes),
            "gt_count": count, "eligible_gt_count": int(eligible.sum()),
            "excluded_gt_count": int((~eligible).sum()), "budgets": budgets,
            "thresholds": [float(t) for t in thresholds],
            "semantics": "many-to-many candidate coverage; not formal matching or AP",
            "ordering": "saved native descending max-logit order; prefixes are not separate Top-K executions",
            "ties_at_prefix_boundaries": [k for k in budgets if 0 < k < len(boxes)
                                           and ranking_scores[k - 1] == ranking_scores[k]],
            "ties_beyond_saved_selection": "unknown",
            "density_definition": f"another eligible GT center within {density_radius:g} original-image diagonals",
            "area_definition": "original-pixel annotation area when supplied, otherwise original-pixel bbox area; small<1024, medium<9216",
            "rows": rows, "per_gt": per_gt}


def coverage_for_image(image, *, budgets=(150, 200, 250, 300), thresholds=(0.3, 0.5, 0.75),
                       density_radius=0.05):
    """Read one saved DETRScope image; no model, SDK or CUDA dependency."""
    encoders = [stage for stage in image["stages"] if stage["source"] == "encoder"]
    if len(encoders) != 1:
        raise ValueError("Expected exactly one selected encoder-proposal stage")
    encoder = encoders[0]
    if encoder.get("box_format") != "normalized_cxcywh" or encoder.get("classification") != "sigmoid":
        raise ValueError("Expected normalized cxcywh proposals and sigmoid logits")
    result = topk_coverage(encoder["boxes"], encoder["logits"], image["targets"],
                           image["metadata"]["original_size"], budgets=budgets,
                           thresholds=thresholds, density_radius=density_radius)
    result["image_id"] = image["metadata"]["image_id"]
    return result
