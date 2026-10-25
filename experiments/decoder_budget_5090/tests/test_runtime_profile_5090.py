"""Reject misleading attribution and archives without requiring server assets or a GPU."""
import copy
import json
import subprocess
import sys

import pytest

from experiments.decoder_budget_5090 import profile_runtime_5090 as runtime
from experiments.decoder_budget_5090 import report_runtime_5090 as report


def event(name, ts, duration, category, **args):
    return {"name":name,"ts":ts,"dur":duration,"ph":"X","cat":category,"tid":1,"args":args}


def test_kernel_correlation_uses_innermost_operator_and_module_once(tmp_path):
    events = [event("iteration::0",0,100,"user_annotation"),
              event("module::model.16:RepC3",1,90,"user_annotation"),
              event("module::model.16.cv1.conv:Conv2d",2,50,"user_annotation"),
              event("aten::convolution",3,40,"cpu_op"), event("aten::cudnn_convolution",4,30,"cpu_op"),
              event("cudaLaunchKernel",10,1,"cuda_runtime",correlation=9),
              event("conv_kernel",20,5,"kernel",correlation=9),
              event("unrelated_background_kernel",200,1000,"kernel",correlation=123)]
    path = tmp_path/"trace.json"
    path.write_text(json.dumps({"traceEvents":events}))
    p = runtime.summarize_trace(path)
    assert p["iterations"][0]["kernel_count"] == 1
    assert p["tables"]["op"][0]["name"] == "aten::cudnn_convolution"
    assert p["tables"]["layer"][0]["name"] == "model.16:RepC3"
    assert p["tables"]["module"][0]["name"] == "model.16.cv1.conv:Conv2d"
    assert p["tables"]["group"][0]["cuda_us_per_forward"] == 5


def test_graph_single_launch_correlates_all_kernels_and_unions_overlap(tmp_path):
    events = [event("iteration::0",0,100,"user_annotation"),
              event("cudaGraphLaunch",10,2,"cuda_runtime",correlation=9),
              event("first",20,10,"kernel",correlation=9),
              event("second",25,10,"kernel",correlation=9),
              event("third",40,5,"kernel",correlation=9)]
    path = tmp_path/"trace.json"
    path.write_text(json.dumps({"traceEvents":events}))
    i = runtime.summarize_trace(path)["iterations"][0]
    assert i["kernel_count"] == 3
    assert i["kernel_sum_us"] == 25
    assert i["gpu_busy_union_us"] == 20
    assert i["gpu_span_us"] == 25
    assert i["gpu_idle_in_span_us"] == 5
    assert i["cpu_launch_gap_us"] is None


def test_missing_cuda_correlation_is_a_failure_not_an_empty_success(tmp_path):
    path = tmp_path/"trace.json"
    path.write_text(json.dumps({"traceEvents":[event("iteration::0",0,100,"user_annotation")]}))
    with pytest.raises(RuntimeError,match="No correlated CUDA"):
        runtime.summarize_trace(path)


@pytest.mark.parametrize("corruption",["truncated", "nan", "summary", "graph", "hook", "kernel_count", "kernel_time", "cpu_fallback"])
def test_archive_rejects_broken_evidence(monkeypatch, corruption):
    original = report.load
    def load(path):
        value = copy.deepcopy(original(path))
        if path.name == "runtime-c2.json":
            c = value["conditions"]["eager_fp32"]
            if corruption == "truncated":
                c["blocks"][0]["cuda_ms"].pop()
            elif corruption == "nan":
                c["blocks"][0]["cuda_ms"][0] = float("nan")
            elif corruption == "summary":
                c["cuda_stats_ms"]["mean"] += .1
            elif corruption == "graph":
                value["conditions"]["graph_fp32"]["changed_input_check"] = False
            elif corruption == "hook":
                value["hooks_remaining"] = 1
        if path.name == "profile-c2-eager_fp32-plain.json":
            if corruption == "kernel_count":
                value["tables"]["op"][0]["calls"] += 1
            elif corruption == "kernel_time":
                value["tables"]["op"][0]["cuda_us"] += 100
        if path.name == "deployment-c2.json" and corruption == "cpu_fallback":
            value["conditions"]["onnx_cuda_fp32"]["backend"]["providers"] = ["CPUExecutionProvider"]
        return value
    monkeypatch.setattr(report,"load",load)
    with pytest.raises(ValueError):
        report.validate(report.ARCHIVE)


def test_real_archive_verifies_with_isolated_standard_library_python():
    script = report.ROOT/"experiments/decoder_budget_5090/report_runtime_5090.py"
    subprocess.run([sys.executable,"-I",str(script),"--verify"],check=True,capture_output=True,text=True)
