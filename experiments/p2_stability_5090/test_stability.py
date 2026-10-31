"""Check diagnostic matching semantics and scratch queue commands on CPU."""
import pytest
from pycocotools.coco import COCO

from experiments.p2_stability_5090.same_gt import attributes, compare, match
from experiments.p2_stability_5090.queue import training_command, validate_training


def fixture_gt():
    gt = COCO()
    gt.dataset = {"images": [{"id": 1, "width": 100, "height": 100}], "categories": [{"id": 1, "name": "object"}],
                  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10], "area": 100, "iscrowd": 0},
                                  {"id": 2, "image_id": 1, "category_id": 1, "bbox": [40, 40, 10, 10], "area": 100, "iscrowd": 0},
                                  {"id": 3, "image_id": 1, "category_id": 1, "bbox": [70, 70, 10, 10], "area": 100, "iscrowd": 1}]}
    gt.createIndex()
    return gt


def test_matching_duplicate_crowd_and_confidence():
    gt = fixture_gt()
    preds = [{"image_id": 1, "category_id": 1, "bbox": box, "score": score}
             for box, score in (([10, 10, 10, 10], .9), ([10, 10, 10, 10], .8),
                                ([40, 40, 10, 10], .02), ([70, 70, 10, 10], .95))]
    states = match(gt, preds)
    assert set(states[.5, 100]) == {1, 2}
    rows = compare({}, states[.5, 100], attributes(gt), .05)
    overall = next(r for r in rows if r["dimension"] == "all")
    assert overall["gt_count"] == 2 and overall["recovered"] == 1 and overall["neither"] == 1


def test_100_and_300_are_per_category_score_prefixes():
    gt = fixture_gt()
    false = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2], "score": .9 - i / 1000} for i in range(100)]
    correct = {"image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10], "score": .1}
    states = match(gt, false + [correct])
    assert not states[.5, 100] and set(states[.5, 300]) == {1}


def test_geometry_threshold_and_empty():
    gt = fixture_gt()
    states = match(gt, [{"image_id": 1, "category_id": 1, "bbox": [12, 10, 10, 10], "score": .9}])
    assert set(states[.5, 100]) == {1} and not states[.75, 100]
    assert not match(gt, [])[.5, 100]


def test_commands_keep_seed_budget_and_scratch_entry(tmp_path):
    command = training_command(tmp_path, "c2", 2)
    assert command[command.index("--seed") + 1] == "2"
    assert command[command.index("--epochs") + 1] == "150"
    assert command[command.index("--fraction") + 1] == "1"
    assert "--resume" not in command
    with pytest.raises(ValueError, match="completed scratch"):
        validate_training({"status": "running"}, "c2", 2)
