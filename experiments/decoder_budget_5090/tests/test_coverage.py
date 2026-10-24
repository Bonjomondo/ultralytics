"""Candidate coverage tests use GT identities, never one-to-one matching."""
import json

import pytest
import torch

from experiments.decoder_budget_5090.coverage import coverage_for_image, topk_coverage


def target(boxes, classes=None, **kwargs):
    boxes = torch.as_tensor(boxes, dtype=torch.float64).reshape(-1, 4)
    return {"boxes": boxes,
            "classes": torch.tensor(classes if classes is not None else [0] * len(boxes), dtype=torch.long),
            "gt_ids": list(range(100, 100 + len(boxes))), **kwargs}


def row(result, budget, group="all", threshold=0.5):
    return next(x for x in result["rows"] if (x["budget"], x["group"], x["iou_threshold"])
                == (budget, group, threshold))


def test_geometry_and_argmax_class_are_separate_with_true_marginal_coverage():
    boxes = torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1]])
    logits = torch.tensor([[5., 0.], [0., 4.], [0., 3.]])
    gt = target(boxes[:2], [1, 1])
    result = topk_coverage(boxes, logits, gt, [1000, 1000], budgets=[0, 1, 2, 3])
    assert row(result, 0)["covered"] == 0
    assert row(result, 1)["covered"] == 1
    assert row(result, 1)["class_correct_covered"] == 0
    assert row(result, 2)["covered"] == 2
    assert row(result, 2)["class_correct_covered"] == 1
    assert row(result, 3)["newly_covered"] == 0
    assert row(result, 3)["newly_class_correct_covered"] == 1
    assert row(result, 3)["class_correct_coverage"] == 1
    assert row(result, 3, "class:0")["coverage"] is None
    assert all(row(result, b)["denominator"] == 2 for b in [0, 1, 2, 3])
    assert next(p for p in result["per_gt"] if p["gt_id"] == 100 and p["iou_threshold"] == .5) == {
        "gt_id": 100, "class_index": 1, "eligible": True, "iou_threshold": .5,
        "first_covered_rank": 1, "first_class_correct_rank": 3}
    json.dumps(result, allow_nan=False)


def test_overlapping_neighbour_gt_are_distinct_and_one_proposal_can_cover_both():
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    gt = target([[0.5, 0.5, 0.2, 0.2], [0.505, 0.5, 0.2, 0.2]])
    result = topk_coverage(boxes, [[3.]], gt, [1000, 1000], budgets=[1])
    assert row(result, 1)["denominator"] == 2
    assert row(result, 1)["covered"] == 2
    assert row(result, 1)["class_correct_covered"] == 2
    assert row(result, 1, "density:nearby")["denominator"] == 2
    assert row(result, 1, "density:isolated")["denominator"] == 0


def test_original_annotation_area_boundaries_and_exclusion_flags():
    boxes = [[.1, .1, .02, .02], [.3, .3, .02, .02], [.5, .5, .02, .02], [.7, .7, .02, .02], [.9, .9, .02, .02]]
    gt = target(boxes, areas=[1023., 1024., 9216., 10., 10.],
                ignore=[False, False, False, True, False], iscrowd=[False, False, False, False, True])
    result = topk_coverage(boxes, [[5.], [4.], [3.], [2.], [1.]], gt, [100, 100], budgets=[5])
    assert result["eligible_gt_count"] == 3
    assert result["excluded_gt_count"] == 2
    assert row(result, 5)["covered"] == 3
    assert row(result, 5, "area:small")["denominator"] == 1
    assert row(result, 5, "area:medium")["denominator"] == 1
    assert row(result, 5, "area:large")["denominator"] == 1
    assert row(result, 5, "density:isolated")["denominator"] == 3
    assert len(result["per_gt"]) == 15


def test_ignored_neighbour_does_not_make_eligible_gt_dense():
    gt = target([[.5, .5, .02, .02], [.5, .5, .02, .02]], ignore=[False, True])
    result = topk_coverage([[.5, .5, .02, .02]], [[1.]], gt, [1000, 1000], budgets=[1])
    assert row(result, 1, "density:nearby")["denominator"] == 0
    assert row(result, 1, "density:isolated")["denominator"] == 1


def test_fallback_area_uses_original_size_not_transformed_size():
    gt = target([[.5, .5, .1, .1]])
    result = topk_coverage([[.5, .5, .1, .1]], [[1.]], gt, [1000, 1000], budgets=[1])
    assert row(result, 1, "area:large")["denominator"] == 1


def test_empty_gt_and_empty_proposals_have_explicit_denominators():
    result = topk_coverage(torch.empty(0, 4), torch.empty(0, 2), target([]), [100, 100], budgets=[0])
    assert result["candidate_count"] == result["eligible_gt_count"] == 0
    assert all(x["denominator"] == 0 and x["coverage"] is None for x in result["rows"])
    assert result["per_gt"] == []
    result = topk_coverage(torch.empty(0, 4), torch.empty(0, 2), target([[.5, .5, .2, .2]]), [100, 100], budgets=[0])
    assert row(result, 0)["denominator"] == 1
    assert row(result, 0)["covered"] == 0
    assert row(result, 0)["coverage"] == 0
    assert all(x["first_covered_rank"] is None for x in result["per_gt"])
    json.dumps(result, allow_nan=False)


def test_tied_native_order_is_preserved_and_flagged_instead_of_resorted():
    gt = target([[.8, .8, .1, .1]])
    boxes = [[.2, .2, .1, .1], [.8, .8, .1, .1]]
    result = topk_coverage(boxes, [[1.], [1.]], gt, [100, 100], budgets=[1, 2])
    assert row(result, 1)["covered"] == 0
    assert row(result, 2)["newly_covered"] == 1
    assert result["ties_at_prefix_boundaries"] == [1]
    assert result["ties_beyond_saved_selection"] == "unknown"
    with pytest.raises(ValueError, match="descending native"):
        topk_coverage(boxes, [[0.], [1.]], gt, [100, 100], budgets=[1])


@pytest.mark.parametrize("budgets", [[2], [-1], [1, 0], [1, 1], [], [True], [1.0]])
def test_invalid_or_unavailable_prefixes_are_rejected(budgets):
    with pytest.raises(ValueError, match="budgets"):
        topk_coverage([[.5, .5, .2, .2]], [[1.]], target([]), [100, 100], budgets=budgets)


@pytest.mark.parametrize("thresholds", [[], [0], [1.1], [float("nan")], [.5, .3], [.5, .5], [True]])
def test_invalid_iou_thresholds_are_rejected(thresholds):
    with pytest.raises(ValueError, match="thresholds"):
        topk_coverage([[.5, .5, .2, .2]], [[1.]], target([]), [100, 100], budgets=[1], thresholds=thresholds)


def test_schema_record_entry_preserves_gt_and_image_id():
    image = {"metadata": {"image_id": 42, "original_size": [100, 100]},
             "targets": target([[.5, .5, .2, .2]]),
             "stages": [{"source": "encoder", "classification": "sigmoid", "box_format": "normalized_cxcywh",
                         "boxes": torch.tensor([[.5, .5, .2, .2]]), "logits": torch.tensor([[1.]])},
                        {"source": "decoder"}]}
    result = coverage_for_image(image, budgets=[1])
    assert result["image_id"] == 42
    assert row(result, 1)["coverage"] == 1
    image["stages"][0]["classification"] = "softmax"
    with pytest.raises(ValueError, match="sigmoid"):
        coverage_for_image(image, budgets=[1])


def test_coverage_monotone_in_budget_and_nonincreasing_in_threshold():
    generator = torch.Generator().manual_seed(42)
    gt_boxes = torch.rand((7, 4), generator=generator) * .5 + .1
    boxes = torch.cat((gt_boxes + .01, gt_boxes), 0)
    logits = torch.arange(len(boxes), 0, -1, dtype=torch.float32).unsqueeze(-1)
    result = topk_coverage(boxes, logits, target(gt_boxes), [1000, 1000], budgets=[0, 2, 5, 10, 14])
    for threshold in [.3, .5, .75]:
        counts = [row(result, k, threshold=threshold)["covered"] for k in [0, 2, 5, 10, 14]]
        assert counts == sorted(counts)
        assert sum(row(result, k, threshold=threshold)["newly_covered"] for k in [0, 2, 5, 10, 14]) == counts[-1]
    for budget in [0, 2, 5, 10, 14]:
        counts = [row(result, budget, threshold=t)["covered"] for t in [.3, .5, .75]]
        assert counts == sorted(counts, reverse=True)
