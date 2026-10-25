"""Gate evidence and lifecycle checks, without scheduling GPU work."""
import argparse
import json

import pytest

from experiments.decoder_budget_5090 import queue_5090 as module


def training(run, *, variant="c0", epochs=1):
    run.mkdir(parents=True)
    (run / "weights").mkdir()
    for name in ("best.pt", "last.pt", "resume.pt"):
        (run / "weights" / name).write_bytes(name.encode())
    module.save(run / "experiment_5090.json", {"status": "completed", "current_epoch": epochs})
    module.save(run / "initialization.json", {"shared_state_verified": True, "shared_state_sha256": "paired"})
    (run / "results.csv").write_text("epoch,train/cls_loss,metrics/mAP50-95(B)\n" +
                                     "".join(f"{i},1.0,0.02\n" for i in range(1, epochs + 1)))
    (run / "health.jsonl").write_text("".join(json.dumps({"epoch": i, "d1_delta_finite": 1.0,
                                                          "p2_first_weight_grad_abs": .001 if variant == "c2" else 0}) + "\n"
                                               for i in range(1, epochs + 1)))


def sweep(run, images=8):
    run.mkdir()
    module.save(run / "summary.json", {"status": "completed", "conditions": [
        {"queries": q, "images": images, "input_gt_identity_sha256": "same"} for q in module.BUDGETS]})


def queue(tmp_path, continuation=False):
    return module.Queue(argparse.Namespace(root=tmp_path / "5090-test", sdk=tmp_path / "sdk",
                                           data_root=tmp_path / "data", gate_only=True,
                                           continue_after_gates=continuation, report_dir=tmp_path / "reports-5090"))


def test_complete_training_and_sweep_receipts(tmp_path):
    run = tmp_path / "5090-c2"
    training(run, variant="c2", epochs=2)
    assert module.validate_training(run, 2, "c2")["shared_state_sha256"] == "paired"
    sweep_run = tmp_path / "5090-query"
    sweep(sweep_run)
    assert module.validate_sweep(sweep_run, 8)["images"] == 8


@pytest.mark.parametrize("failure", ["epochs", "health", "gradient", "checkpoint"])
def test_training_rejects_incomplete_or_unhealthy_evidence(tmp_path, failure):
    run = tmp_path / "5090-c2"
    training(run, variant="c2")
    if failure == "epochs":
        module.save(run / "experiment_5090.json", {"status": "completed", "current_epoch": 0})
    elif failure == "health":
        (run / "health.jsonl").write_text('{"epoch":1,"d1_delta_finite":0}\n')
    elif failure == "gradient":
        (run / "health.jsonl").write_text('{"epoch":1,"d1_delta_finite":1,"p2_first_weight_grad_abs":0}\n')
    else:
        (run / "weights/resume.pt").unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        module.validate_training(run, 1, "c2")


def test_sweep_rejects_input_identity_drift(tmp_path):
    run = tmp_path / "5090-query"
    sweep(run)
    result = module.read(run / "summary.json")
    result["conditions"][-1]["input_gt_identity_sha256"] = "different"
    module.save(run / "summary.json", result)
    with pytest.raises(RuntimeError, match="identity"):
        module.validate_sweep(run, 8)


def test_phase_never_reuses_output_or_overwrites_logs(tmp_path, monkeypatch):
    instance = queue(tmp_path)
    instance.root.mkdir()
    instance.state = {"status": "gates_passed", "completed_phases": []}
    monkeypatch.setattr(instance, "check_sources", lambda: None)
    output = instance.root / "5090-existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        instance.phase("5090-existing", ["never-execute"], output)
    assert not instance.status.exists()
    output.rmdir()
    logs = instance.root / "logs_5090"
    logs.mkdir()
    log = logs / "5090-existing.log"
    log.write_text("old evidence")
    with pytest.raises(FileExistsError):
        instance.phase("5090-existing", ["never-execute"], output)
    assert log.read_text() == "old evidence"


def test_continuation_rejects_tampered_checkpoint_without_mutating_old_state(tmp_path, monkeypatch):
    instance = queue(tmp_path, continuation=True)
    instance.root.mkdir()
    original_status = {"status": "gates_passed", "completed_phases": []}
    module.save(instance.status, original_status)
    instance.pin.write_text("pin")
    for variant in ("c0", "c2"):
        run = instance.root / f"5090-rtdetr-r18-{variant}-1e-gate-s0"
        training(run, variant=variant)
        sweep(instance.root / (run.name + "-query-gate"))
    train_receipts, sweep_receipts = instance.verify_gates()
    module.save(instance.gates, {"source_pin_sha256": module.digest(instance.pin),
                                "training": train_receipts, "sweeps": sweep_receipts})
    (instance.root / "5090-rtdetr-r18-c2-1e-gate-s0/weights/best.pt").write_bytes(b"tampered")
    monkeypatch.setattr(instance, "check_sources", lambda: None)
    with pytest.raises(RuntimeError, match="evidence changed"):
        instance.run()
    assert module.read(instance.status) == original_status


def test_failed_new_queue_retains_failure_state(tmp_path, monkeypatch):
    instance = queue(tmp_path)
    def failing_prepare():
        instance.root.mkdir()
        instance.owned = True
        raise RuntimeError("deliberate preparation failure")
    monkeypatch.setattr(instance, "prepare", failing_prepare)
    with pytest.raises(RuntimeError, match="deliberate"):
        instance.run()
    assert module.read(instance.status)["status"] == "failed"


def test_root_name_requires_5090(tmp_path):
    with pytest.raises(ValueError, match="5090"):
        module.Queue(argparse.Namespace(root=tmp_path / "old", sdk=tmp_path / "sdk", data_root=tmp_path))
