"""Lifecycle and end-to-end checks of the real native query-budget entry point."""
import argparse
import json
import os
from pathlib import Path
import sys

import pytest
import torch

from experiments.decoder_budget_5090 import run


def test_query_budget_restores_after_exception_and_rejects_learned_queries():
    head = argparse.Namespace(num_queries=300, learnt_init_query=False)
    with pytest.raises(RuntimeError, match="deliberate"):
        with run.query_budget(head, 150):
            assert head.num_queries == 150
            raise RuntimeError("deliberate")
    assert head.num_queries == 300
    head.learnt_init_query = True
    with pytest.raises(ValueError, match="encoder-initialized"):
        with run.query_budget(head, 150):
            pass
    assert head.num_queries == 300


def test_coverage_pools_gt_denominators():
    def image(denominator, covered):
        return {"rows": [{"budget": 150, "iou_threshold": .5, "group": "all",
                          "denominator": denominator, "covered": covered,
                          "class_correct_covered": covered, "newly_covered": covered,
                          "newly_class_correct_covered": covered}]}
    row = run.summarize_coverage([image(1, 1), image(9, 0)])["rows"][0]
    assert row["denominator"] == 10 and row["coverage"] == .1
    assert run.summarize_coverage([image(0, 0)])["rows"][0]["coverage"] is None


def test_main_never_modifies_existing_failed_evidence(tmp_path, monkeypatch):
    (tmp_path / "failure.json").write_text('{"old": true}\n')
    original = (tmp_path / "failure.json").read_bytes()
    monkeypatch.setattr(sys, "argv", ["run.py", "smoke", "--pin", "missing-pin.json", "--output", str(tmp_path)])
    with pytest.raises(FileExistsError):
        run.main()
    assert (tmp_path / "failure.json").read_bytes() == original
    assert set(p.name for p in tmp_path.iterdir()) == {"failure.json"}


def test_main_marks_own_failed_progress(tmp_path, monkeypatch):
    destination = tmp_path / "new"
    def fail(args):
        args.output.mkdir()
        args._created_output = args.output.resolve()
        run.write(args.output / "progress.json", {"status": "running", "conditions": [{"queries": 300}]})
        raise RuntimeError("deliberate")
    monkeypatch.setattr(run, "smoke", fail)
    monkeypatch.setattr(sys, "argv", ["run.py", "smoke", "--pin", "missing-pin.json", "--output", str(destination)])
    with pytest.raises(RuntimeError, match="deliberate"):
        run.main()
    progress = json.loads((destination / "progress.json").read_text())
    assert progress["status"] == "failed" and progress["conditions"] == [{"queries": 300}]


def test_synthetic_checkpoint_full_sweep_uses_native_decoder_and_coco(tmp_path):
    import cv2
    import numpy as np
    from experiments.decoder_budget_5090.source_adapter import create_source_pin
    from detrscope.core.records import iter_batches
    run.configure()
    model = run.tiny_model()
    checkpoint = tmp_path / "tiny.pt"
    torch.save({"model": model}, checkpoint)
    pin = tmp_path / "pin.json"
    create_source_pin(run.ROOT, pin, expected_commit="6d1d4d1badbecf47970243685a1d712ad3e31559")
    images = tmp_path / "images"
    images.mkdir()
    for i in (1, 2):
        assert cv2.imwrite(str(images / f"{i}.png"), np.full((48, 64, 3), i * 71, dtype=np.uint8))
    annotations = tmp_path / "annotations.json"
    run.write(annotations, {"images": [{"id": i, "width": 64, "height": 48, "file_name": f"{i}.png"} for i in (1, 2)],
                            "categories": [{"id": i, "name": name} for i, name in ((3, "person"), (7, "car"), (9, "bike"))],
                            "annotations": [{"id": i, "image_id": i, "category_id": 3, "bbox": [8, 8, 16, 16],
                                             "area": 256, "iscrowd": 0} for i in (1, 2)]})
    destination = tmp_path / "sweep"
    device = os.environ.get("DECODER_BUDGET_TEST_DEVICE", "cpu")
    args = argparse.Namespace(sdk=run.ROOT.parent / "detrscope", output=destination, device=device, expected_gpu="5090",
                              checkpoint=checkpoint, annotations=annotations, images=images, pin=pin,
                              budgets=[300, 150], limit=0, batch=2, imgsz=32, conf=.001, benchmark=device.startswith("cuda"))
    run.sweep(args)
    result = json.loads((destination / "summary.json").read_text())
    assert result["status"] == "completed"
    assert result["conditions"][0]["input_gt_identity_sha256"] == result["conditions"][1]["input_gt_identity_sha256"]
    for budget in (300, 150):
        raw = destination / f"q{budget}"
        records = [im for payload in iter_batches(raw) for im in payload["images"]]
        assert len(records) == 2
        assert all(len(s["boxes"]) == budget for im in records for s in im["stages"])
        predictions = json.loads((raw / "predictions.json").read_text())
        assert len(predictions) == budget * 2
        assert {p["category_id"] for p in predictions} <= {3, 7, 9}
        assert json.loads((raw / "coco.json").read_text())["protocol"]["image_ids"] == [1, 2]
        assert (raw / "coverage_summary.csv").is_file() and (raw / "report.html").is_file()
        if device.startswith("cuda"):
            timing = json.loads((raw / "benchmark.json").read_text())
            assert len(timing["latency_ms"]) == 200 and timing["median_ms"] > 0
    assert json.loads((destination / "q150/formal_gt_changes.json").read_text())["schema"] == "detrscope.gt-comparison.v1"
