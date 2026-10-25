"""Sampling must change real dimensions; native early exits must skip later work."""
import copy
import subprocess
import sys
import json

import pytest
import torch

from experiments.decoder_budget_5090.run import tiny_model
from experiments.decoder_budget_5090.screen_decoder_perf_5090 import PredictionOnly, replace_sampling
from experiments.decoder_budget_5090.screen_decoder_eval_5090 import pooled_formal_changes
from experiments.decoder_budget_5090.screen_decoder_eval_5090 import write as write_evaluation
from experiments.decoder_budget_5090 import report_decoder_screen_5090 as report


@pytest.mark.parametrize("points",[1,2,4])
def test_point_budgets_change_dimensions_and_keep_common_projection_bytes(points):
    a,b = tiny_model(),tiny_model()
    replace_sampling(a,4); inventory = replace_sampling(b,points)
    for index,(left,right) in enumerate(zip(a.model[-1].decoder.layers,b.model[-1].decoder.layers)):
        x,y = left.cross_attn,right.cross_attn
        assert y.sampling_offsets.out_features == 4*3*points*2
        assert y.attention_weights.out_features == 4*3*points
        assert inventory[index]["parameters"] == 2*16*17+3*17*4*3*points
        assert torch.equal(x.value_proj.weight,y.value_proj.weight)
        assert torch.equal(x.output_proj.weight,y.output_proj.weight)
        expected = x.sampling_offsets.weight.view(4,3,4,2,16)[:,:,:points].reshape(-1,16)
        assert torch.equal(expected,y.sampling_offsets.weight)
        assert y.sampling_offsets.weight.abs().sum()>0
        assert y.attention_weights.weight.abs().sum()>0


@pytest.mark.parametrize("depth",[1,2,3])
def test_native_early_exit_really_skips_later_layers_and_changes_query_count(depth):
    model = tiny_model()
    head = model.model[-1]
    head.num_queries = 200
    head.decoder.eval_idx = depth-1
    seen,handles = [],[]
    for i,layer in enumerate(head.decoder.layers):
        handles.append(layer.register_forward_hook(lambda m,a,y,i=i:seen.append(i+1)))
    try:
        with torch.inference_mode():
            pred = model(torch.rand(1,3,64,64))[0]
    finally:
        for h in handles:
            h.remove()
    assert seen == list(range(1,depth+1))
    assert pred.shape == (1,200,7)
    assert torch.isfinite(pred).all()


def test_export_wrapper_eval_is_restored_without_switching_network_to_train(tmp_path):
    model = tiny_model()
    wrapper = PredictionOnly(model).eval()
    fast = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.inference_mode():
            torch.onnx.export(wrapper,torch.rand(1,3,64,64),str(tmp_path/"tiny.onnx"),opset_version=17,dynamo=False)
    finally:
        torch.backends.mha.set_fastpath_enabled(fast)
    assert not any(m.training for m in wrapper.modules())
    with torch.inference_mode():
        assert wrapper(torch.rand(1,3,64,64)).shape == (1,300,7)


def test_formal_gt_changes_pool_same_gt_and_stratify_area_category():
    baseline = {"schema":"detrscope.coco.v1","protocol":{"iou_thresholds":[.5]},
                "gt_states":{"10":{"image_id":1,"category_id":1,"matched":[True]},
                             "20":{"image_id":1,"category_id":2,"matched":[False]}}}
    candidate = copy.deepcopy(baseline)
    candidate["gt_states"]["10"]["matched"] = [False]
    candidate["gt_states"]["20"]["matched"] = [True]
    annotations = {"10":{"category_id":1,"area":100},"20":{"category_id":2,"area":2000}}
    _,rows = pooled_formal_changes(baseline,candidate,annotations,"D2->D3")
    all_row = next(r for r in rows if r["group"]=="all")
    assert all_row["gt_denominator"] == 2
    assert all_row["lost"] == all_row["gained"] == 1
    assert all_row["before_tp"] == all_row["after_tp"] == 1
    assert next(r for r in rows if r["group"]=="size:small")["lost"] == 1
    candidate["gt_states"]["20"]["image_id"] = 2
    with pytest.raises(ValueError,match="metadata differs"):
        pooled_formal_changes(baseline,candidate,annotations,"D2->D3")


def test_coco_result_preserves_numpy_gt_id_types_as_json_numbers(tmp_path):
    import numpy as np
    path = tmp_path/"formal.json"
    write_evaluation(path,{"gt_states":{"10":{"image_id":np.int64(3),"category_id":np.int64(2),"matched":[True]}}})
    result = json.loads(path.read_text())
    assert result["gt_states"]["10"] == {"image_id":3,"category_id":2,"matched":[True]}


def test_reduction_uses_paired_rounds_and_rejects_missing_pairs():
    before = {"blocks":[{"round":i,"cuda_ms":[10.,10.]} for i in range(1,11)],"cuda_stats_ms":{"mean":10.}}
    after = {"blocks":[{"round":i,"cuda_ms":[9.,9.]} for i in range(1,11)],"cuda_stats_ms":{"mean":9.}}
    r = report.reduction(before,after)
    assert r["latency_reduction_percent"] == pytest.approx(10.)
    assert r["ci95_low"] == pytest.approx(10.)
    assert r["ci95_high"] == pytest.approx(10.)
    assert r["paired_rounds"] == 10
    after["blocks"].pop()
    with pytest.raises(ValueError,match="Unpaired"):
        report.reduction(before,after)


@pytest.mark.parametrize("corruption",["nan","duplicate_round","masked_points","missing_engine_input","checkpoint","early_exit"])
def test_compact_archive_rejects_invalid_measurements(monkeypatch,corruption):
    original = report.load
    def altered(path):
        data = copy.deepcopy(original(path))
        condition = data["performance"][("sampling","c2","model")]["q300-p2"]["trt_fp16"]
        if corruption=="nan":
            condition["blocks"][0]["cuda_ms"][0] = float("nan")
        elif corruption=="duplicate_round":
            condition["blocks"][0]["round"] = 2
        elif corruption=="masked_points":
            data["identity"][("sampling","c2")]["conditions"]["q300-p2"]["attention_inventory"][0]["offset_outputs"] *= 2
        elif corruption=="missing_engine_input":
            data["performance"][("sampling","c2","attention")]["q300-p2"]["trt_fp16"]["engine"]["input_names"].pop()
        elif corruption=="checkpoint":
            data["evaluation"]["c2"]["checkpoint"]["sha256"] = "other"
        elif corruption=="early_exit":
            data["evaluation"]["c2"]["early_exit_equivalence"]["all_exact"] = False
        return data
    monkeypatch.setattr(report,"load",altered)
    with pytest.raises(ValueError):
        report.validate(report.ARCHIVE)


def test_screen_archive_verifies_without_importing_torch_or_server_assets():
    script = report.ROOT/"experiments/decoder_budget_5090/report_decoder_screen_5090.py"
    subprocess.run([sys.executable,"-I",str(script),"--verify"],check=True,capture_output=True,text=True)
