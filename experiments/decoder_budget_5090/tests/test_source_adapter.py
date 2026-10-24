"""CPU integration checks against this fork's real native DetectionModel."""
import json
from pathlib import Path
import subprocess

import pytest
import torch

from detrscope import Debugger, DetectionBatch
from detrscope.adapters.base import UnsupportedCapability
from detrscope.adapters.ultralytics_rtdetr import UltralyticsRTDETRAdapter
from detrscope.core.identity import assert_outputs_equal
from detrscope.observers import StageRecorder
from experiments.decoder_budget_5090.source_adapter import PinnedRTDETRAdapter, create_source_pin


REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def pin(tmp_path_factory):
    path = tmp_path_factory.mktemp("source-pin") / "pin.json"
    create_source_pin(REPO, path, expected_commit=subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())
    return path


def model_and_batch(*, queries=7, eval_idx=-1, learnt=False, device="cpu"):
    from ultralytics.nn.tasks import RTDETRDetectionModel
    torch.set_num_threads(1)
    torch.manual_seed(53)
    configuration = {
        "nc": 3,
        "backbone": [[-1, 1, "Conv", [16, 3, 2]], [-1, 1, "Conv", [16, 3, 2]],
                     [-1, 1, "Conv", [16, 3, 2]]],
        "head": [[[0, 1, 2], 1, "RTDETRDecoder", [3, 16, queries, 4, 4, 3, 32]]],
    }
    model = RTDETRDetectionModel(cfg=configuration, ch=3, nc=3, verbose=False).to(device).eval()
    model.model[-1].decoder.eval_idx = 2 if eval_idx == -1 else eval_idx
    model.model[-1].learnt_init_query = learnt
    with torch.no_grad():
        for head in model.model[-1].dec_bbox_head:
            head.layers[-1].weight.normal_(std=0.05)
    inputs = torch.rand(2, 3, 32, 32, device=device)
    metadata = [{"image_id": index, "original_size": [80, 120], "transformed_size": [32, 32],
                 "normalization": {"transform": "stretch"}} for index in range(2)]
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]), "classes": torch.tensor([0]),
                "gt_ids": [9 + index]} for index in range(2)]
    return model, DetectionBatch(inputs, metadata, targets)


def adapter(model, pin):
    return PinnedRTDETRAdapter(model, repo=REPO, pin=pin,
                               class_names=["person", "car", "bike"], category_ids=[1, 2, 3])


def hooks(model):
    return sum(len(module._forward_hooks) + len(module._forward_pre_hooks) for module in model.modules())


@pytest.mark.parametrize("queries,eval_idx", [(7, 2), (4, 2), (7, 0)])
def test_native_hooks_exact_output_rng_probabilities_and_query_identity(pin, queries, eval_idx):
    model, batch = model_and_batch(queries=queries, eval_idx=eval_idx)
    plugin = adapter(model, pin)
    recorder = StageRecorder()
    recorder.model_metadata = plugin.describe()
    recorder.requested_probes = {"attention", "deformable_sampling"}
    recorder.begin_batch(batch)
    with torch.inference_mode():
        before = torch.random.get_rng_state().clone()
        native = model(batch.inputs)
        native_rng = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(before)
        with plugin.observe(recorder):
            observed = model(batch.inputs)
        assert torch.equal(torch.random.get_rng_state(), native_rng)
        restored = model(batch.inputs)
        images = recorder.finish_batch()
        plugin.validate_observation(observed, images)
    assert_outputs_equal(native, observed)
    assert_outputs_equal(native, restored)
    assert hooks(model) == 0 and plugin._observing is False
    assert plugin.describe()["upstream"]["wheel_record_verified"] is False
    assert plugin.describe()["normal_queries"] == queries
    assert plugin.describe()["executed_eval_layer_ids"] == list(range(1, eval_idx + 2))
    for image_index, image in enumerate(images):
        encoder, *decoders = image["stages"]
        assert encoder["source"] == "encoder"
        assert torch.equal(encoder["decoder_query_ids"], torch.arange(queries))
        # Native encoder token identities are separate from rank/decoder slots.
        assert not torch.equal(encoder["query_ids"], encoder["decoder_query_ids"])
        assert [stage["layer_id"] for stage in decoders] == list(range(1, eval_idx + 2))
        for stage in decoders:
            assert torch.equal(stage["query_ids"], torch.arange(queries))
        assert torch.equal(decoders[-1]["probabilities"], native[1][1].squeeze(0).sigmoid()[image_index])
        assert torch.equal(decoders[-1]["boxes"], native[1][0].squeeze(0)[image_index])
        assert len(image["probes"]) == 2 * (eval_idx + 1)


def test_changed_query_count_reexecutes_every_decoder_layer(pin):
    model, batch = model_and_batch(queries=7)
    plugin = adapter(model, pin)
    observed_counts = []
    tap = model.model[-1].decoder.layers[0].register_forward_pre_hook(
        lambda module, args: observed_counts.append(int(args[0].shape[1])))
    outputs = []
    try:
        with torch.inference_mode():
            for count in (7, 4):
                model.model[-1].num_queries = count
                recorder = StageRecorder()
                recorder.model_metadata = plugin.describe()
                recorder.begin_batch(batch)
                with plugin.observe(recorder):
                    result = model(batch.inputs)
                plugin.validate_observation(result, recorder.finish_batch())
                outputs.append(result[0].detach().clone())
    finally:
        tap.remove()
    assert observed_counts == [7, 4]
    assert [output.shape[1] for output in outputs] == [7, 4]
    # Self-attention is actually rerun with a new budget; this is not a slice.
    assert not torch.equal(outputs[0][:, :4], outputs[1])
    assert hooks(model) == 0


def test_debugger_native_identity_and_offline_lifecycle(pin, tmp_path):
    model, batch = model_and_batch()
    debugger = Debugger(model, adapter(model, pin), tmp_path / "recording")
    result = debugger.observe([batch], probes="auto")
    assert result["status"] == "completed"
    assert result["batches"][0]["identity_passed"] is True
    assert result["batches"][0]["restoration_passed"] is True
    assert result["model"]["upstream"]["sdk_source_sha256"]
    assert debugger.analyze()["image_count"] == 2
    assert debugger.report().is_file()
    assert hooks(model) == 0


def test_exception_uninstalls_all_hooks_and_restores_native(pin):
    model, batch = model_and_batch()
    plugin = adapter(model, pin)
    recorder = StageRecorder()
    recorder.begin_batch(batch)
    recorder.requested_probes = {"attention", "deformable_sampling"}
    def fail(module, args):
        raise RuntimeError("deliberate native decoder failure")
    failure = model.model[-1].decoder.layers[1].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="deliberate"), torch.inference_mode(), plugin.observe(recorder):
            model(batch.inputs)
        assert hooks(model) == 1  # only the caller's deliberate failure hook
        assert plugin._observing is False
    finally:
        failure.remove()
    with torch.inference_mode():
        native = model(batch.inputs)
    assert native[0].shape[1] == 7 and hooks(model) == 0


def test_learned_queries_and_instance_forward_overrides_rejected(pin):
    model, _ = model_and_batch(learnt=True)
    with pytest.raises(UnsupportedCapability, match="learnt_init_query=False"):
        adapter(model, pin)
    model, _ = model_and_batch()
    model.model[-1].forward = model.model[-1].forward
    with pytest.raises(UnsupportedCapability, match="per-instance forward"):
        adapter(model, pin)


def test_pin_is_explicit_and_refuses_overwrite(tmp_path):
    output = tmp_path / "pin.json"
    with pytest.raises(ValueError, match="expected_commit"):
        create_source_pin(REPO, output, expected_commit="")
    create_source_pin(REPO, output, expected_commit="HEAD")
    with pytest.raises(FileExistsError):
        create_source_pin(REPO, output, expected_commit="HEAD")


@pytest.mark.parametrize("mutation", ["source", "sdk", "location", "ancestor"])
def test_tampered_pin_and_wrong_ancestor_rejected(pin, tmp_path, mutation):
    content = json.loads(pin.read_text())
    if mutation == "source":
        content["source_sha256"][next(iter(content["source_sha256"]))] = "0" * 64
    elif mutation == "sdk":
        content["sdk_source_sha256"][next(iter(content["sdk_source_sha256"]))] = "0" * 64
    elif mutation == "location":
        content["repo_root"] = str(tmp_path)
    else:
        content["expected_ancestor_commit"] = "0" * 40
    changed = tmp_path / "changed-pin.json"
    changed.write_text(json.dumps(content))
    model, _ = model_and_batch()
    with pytest.raises(UnsupportedCapability):
        adapter(model, changed)


def test_on_disk_added_source_rejected_before_model_import(tmp_path):
    repo = tmp_path / "isolated-source"
    (repo / "ultralytics").mkdir(parents=True)
    (repo / "ultralytics" / "__init__.py").write_text("# explicitly pinned fixture\n")
    for args in (["init", "-q"], ["add", "ultralytics"],
                 ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    output = tmp_path / "isolated-pin.json"
    create_source_pin(repo, output, expected_commit="HEAD")
    (repo / "ultralytics" / "new_unreviewed.py").write_text("CHANGED = True\n")
    with pytest.raises(UnsupportedCapability, match="source tree differs"):
        PinnedRTDETRAdapter(torch.nn.Linear(1, 1), repo=repo, pin=output,
                            class_names=["person"], category_ids=[1])


def test_pin_and_runtime_changes_after_construction_rejected(pin, tmp_path):
    copied = tmp_path / "pin.json"
    copied.write_bytes(pin.read_bytes())
    model, batch = model_and_batch()
    plugin = adapter(model, copied)
    copied.write_text(copied.read_text() + "\n")
    recorder = StageRecorder()
    recorder.begin_batch(batch)
    with pytest.raises(UnsupportedCapability, match="pin changed"), plugin.observe(recorder):
        pass
    plugin = adapter(model, pin)
    model.model[-1].learnt_init_query = True
    with pytest.raises(UnsupportedCapability, match="learnt_init_query=False"), plugin.observe(recorder):
        pass
    assert hooks(model) == 0


def test_official_wheel_gate_still_rejects_this_source_fork():
    model, _ = model_and_batch()
    with pytest.raises(UnsupportedCapability):
        UltralyticsRTDETRAdapter(model)
