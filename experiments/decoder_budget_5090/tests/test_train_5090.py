"""Training lifecycle gates, without launching a GPU or actual training."""
import argparse
import json

import pytest
import yaml

from experiments.decoder_budget_5090 import train_5090


def arguments(tmp_path):
    data = tmp_path / "5090-data.yaml"
    data.write_text(yaml.safe_dump({"path": str(tmp_path), "train": "images/train", "val": "images/val",
                                    "nc": 10, "names": [f"class{i}" for i in range(10)]}))
    return argparse.Namespace(variant="c0", data=data, project=tmp_path / "5090-experiments",
                              name="5090-rtdetr-r18-c0-smoke", epochs=1, fraction=.01, device="0")


@pytest.mark.parametrize("environment", [{"cuda_available": False}, {"cuda_available": True, "gpu": "RTX 4090"}])
def test_unavailable_or_wrong_gpu_records_failure_without_starting_training(tmp_path, monkeypatch, environment):
    args = arguments(tmp_path)
    monkeypatch.setattr(train_5090, "runtime_environment", lambda: environment)
    monkeypatch.setattr(train_5090, "build_model", lambda *a, **k: pytest.fail("GPU gate must run before building models"))
    with pytest.raises(RuntimeError, match="5090"):
        train_5090.train(args)
    result = json.loads((args.project / args.name / "experiment_5090.json").read_text())
    assert result["status"] == "failed" and result["current_epoch"] == 0
    assert result["environment"] == environment and result["source_sha256"]


def test_second_launch_cannot_overwrite_existing_run(tmp_path):
    args = arguments(tmp_path)
    run = args.project / args.name
    run.mkdir(parents=True)
    evidence = run / "experiment_5090.json"
    evidence.write_text('{"status":"running","original":true}\n')
    original = evidence.read_bytes()
    with pytest.raises(FileExistsError):
        train_5090.train(args)
    assert evidence.read_bytes() == original
    assert [item.name for item in run.iterdir()] == [evidence.name]


def test_namespace_and_explicit_data_gate_precede_creating_output(tmp_path):
    args = arguments(tmp_path)
    args.name = "historical-c0"
    with pytest.raises(ValueError, match="start with 5090"):
        train_5090.train(args)
    assert not args.project.exists()
    args.name = "5090-c0"
    args.data = train_5090.ROOT / "configs/visdrone-local.yaml"
    with pytest.raises(ValueError, match="explicit 5090 data YAML"):
        train_5090.train(args)
    assert not args.project.exists()


def test_health_rejects_nonfinite_and_null_measurements():
    train_5090.finite_numbers({"gamma": None, "encoder_delta_finite": 1.0}, location="health")
    with pytest.raises(FloatingPointError):
        train_5090.finite_numbers({"encoder_delta_query_variance": None}, location="health")
    with pytest.raises(FloatingPointError):
        train_5090.finite_numbers({"train_seconds": float("nan")}, location="health")
