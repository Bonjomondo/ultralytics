"""CPU checks for factorial fairness, actual decoder/loss integration and report gates."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.p2_p5_la_5090.model import MODELS, FactorialTrainer, build_model, identify, inspect_model
from experiments.p2_p5_la_5090.report import ARMS, validate
from experiments.r18_p2_sfi.train import build_model as historical_model, tensor_digest
from ultralytics import RTDETR


@pytest.fixture(scope="module")
def models():
    torch.set_num_threads(2)
    return {name: build_model(str(cfg)) for name, cfg in MODELS.items()}


def test_factorial_changes_only_p5_and_p2_routes(models):
    for arm, model in models.items():
        inspect_model(model, arm)
        assert model.yaml["backbone"] == models["c0"].yaml["backbone"]
        expected = deepcopy(models["c2" if arm in ("c2", "c2-la") else "c0"].yaml["head"])
        if arm in ("la", "c2-la"):
            expected[1] = models["la"].yaml["head"][1]
        assert model.yaml["head"] == expected
    bad = deepcopy(models["c2-la"].yaml)
    bad["head"][-1][-1][3] = 2
    with pytest.raises(ValueError):
        identify(bad)


def test_shared_and_branch_specific_initialization(models):
    records = {key: model.ablation_initialization for key, model in models.items()}
    assert len({item["shared_state_sha256"] for item in records.values()}) == 1
    for slot, pairs in ((6, (("c0", "c2"), ("la", "c2-la"))), (14, (("c0", "la"), ("c2", "c2-la")))):
        for left, right in pairs:
            assert tensor_digest(models[left].model[slot].state_dict()) == tensor_digest(models[right].model[slot].state_dict())
    for arm in ("c0", "c2"):
        old = historical_model(str(MODELS[arm]))
        assert tensor_digest(old.state_dict()) == tensor_digest(models[arm].state_dict())


def test_actual_trainer_rebuild_uses_paired_model(models):
    trainer = FactorialTrainer.__new__(FactorialTrainer)
    trainer.data = {"nc": 10, "channels": 3}
    trainer.args = SimpleNamespace(seed=0)
    rebuilt = trainer.get_model(cfg=deepcopy(models["c2-la"].yaml), verbose=False)
    assert tensor_digest(rebuilt.state_dict()) == tensor_digest(models["c2-la"].state_dict())
    with pytest.raises(ValueError, match="from scratch"):
        trainer.get_model(cfg=rebuilt.yaml, weights=rebuilt)


def test_joint_full_forward_loss_gradients_and_checkpoint(models, tmp_path):
    model = deepcopy(models["c2-la"]).eval()
    inputs = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        expected = model(inputs)[0]
    assert expected.shape == (2, 300, 14)
    path = tmp_path / "joint.pt"
    torch.save({"model": model}, path)
    restored = RTDETR(str(path)).model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs)[0], expected, atol=0, rtol=0)
    model.train()
    loss, parts = model({"img": inputs, "batch_idx": torch.tensor([0, 1]),
                         "cls": torch.tensor([[0.], [3.]]),
                         "bboxes": torch.tensor([[.3, .4, .1, .2], [.6, .5, .2, .1]])})
    assert torch.isfinite(loss).all() and torch.isfinite(parts).all()
    loss.sum().backward()
    for parameter in (next(model.model[14].parameters()), model.model[6].attn.qkv.weight,
                      model.model[6].attn.relative_position_bias):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


def test_coco_semantic_mapping_and_exact_detection(tmp_path):
    from pycocotools.coco import COCO
    from experiments.p2_p5_la_5090.evaluate import category_ids, coco_metrics
    from experiments.p2_p5_la_5090.train import NAMES
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({"info": {}, "images": [{"id": 2, "file_name": "a.jpg", "width": 64, "height": 64}],
                               "categories": [{"id": i + 20, "name": name} for i, name in enumerate(NAMES)],
                               "annotations": [{"id": 1, "image_id": 2, "category_id": 23, "bbox": [10, 10, 20, 20],
                                                "area": 400, "iscrowd": 0}]}))
    gt = COCO(str(path))
    assert category_ids(gt, NAMES) == list(range(20, 30))
    metrics, classes = coco_metrics(gt, [{"image_id": 2, "category_id": 23, "bbox": [10, 10, 20, 20], "score": .9}], [2])
    assert metrics["AP"] > .99 and metrics["APs"] > .99 and classes["car"]["AP"] > .99
    empty, _ = coco_metrics(gt, [], [2])
    assert empty["AP"] == 0


def test_report_rejects_unpaired_or_subset_results():
    evaluations = {arm: {"status": "completed", "variant": arm, "scope": "full_val", "training_role": "fixed_budget_screen",
                         "training_fraction": 1, "protocol": {}, "training_recipe": {}, "training_source_sha256": {},
                         "data_yaml_sha256": "data", "initialization": {"shared_state_sha256": "shared",
                            "p2_slot_sha256": "p2", "p5_slot_sha256": "p5"},
                         "seed": 0, "training_epochs": 150, "checkpoint": {"sha256": arm}} for arm in ARMS}
    benchmark = {"status": "completed", "role": "trained_checkpoint_comparison",
                 "models": {arm: {"identity": {"sha256": arm}} for arm in ARMS}}
    validate(evaluations, benchmark)
    for key, value in (("scope", "subset_gate"), ("seed", 1), ("training_epochs", 350)):
        changed = deepcopy(evaluations)
        changed["c2-la"][key] = value
        with pytest.raises(ValueError):
            validate(changed, benchmark)
    changed = deepcopy(benchmark)
    changed["models"]["la"]["identity"]["sha256"] = "wrong"
    with pytest.raises(ValueError):
        validate(evaluations, changed)


def test_training_entry_health_and_final_checkpoints_on_synthetic_cpu(tmp_path, monkeypatch):
    """Exercise callbacks and completion checks; CPU/128 override is engineering QA only."""
    import cv2
    import numpy as np
    import yaml
    from experiments.p2_p5_la_5090 import train as entry
    for split, count in (("train", 8), ("val", 2)):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        for i in range(count):
            image = np.random.default_rng(i).integers(0, 256, (128, 128, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / "images" / split / f"{i}.jpg"), image)
            (tmp_path / "labels" / split / f"{i}.txt").write_text(f"{i % 10} 0.5 0.5 0.2 0.2\n")
    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(tmp_path), "train": "images/train", "val": "images/val",
                                   "names": entry.NAMES, "nc": 10}))
    args = entry.parse_args(["--variant", "c2-la", "--data", str(data), "--project", str(tmp_path / "runs"),
                             "--name", "5090-cpu-smoke", "--epochs", "1"])
    native_train = RTDETR.train

    def cpu_train(self, **kwargs):
        kwargs.update(device="cpu", imgsz=128, plots=False)
        return native_train(self, **kwargs)

    monkeypatch.setattr(entry, "configure", lambda: None)
    monkeypatch.setattr(RTDETR, "train", cpu_train)
    entry.train(args)
    run = args.project / args.name
    state = json.loads((run / "experiment.json").read_text())
    assert state["status"] == "completed" and state["role"] == "engineering_smoke"
    assert state["effective_args"]["device"] == "cpu" and state["effective_args"]["imgsz"] == 128
    assert set(state["checkpoints"]) == {"best", "last", "resume"}
    assert state["initialization"]["shared_state_verified"]
