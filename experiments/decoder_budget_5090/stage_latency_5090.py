"""Validate archived within-forward timings and expose a compact offline summary."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = "docs/experiment_reports/2026-10-24_5090_baselines/raw/stage_latency_5090.json"
IDENTITY = "docs/experiment_reports/2026-10-24_5090_baselines/raw/stage_latency_identity_5090.json"
STAGES = ("backbone_0_to_4", "neck_encoder_5_to_23", "input_projection_and_query_selection",
          "decoder_and_prediction_heads", "output_assembly")
LABELS = ("Backbone（层0–4）", "Neck/Encoder（层5–23）", "Head输入投影与Query选择", "Decoder与预测头", "输出拼装")
FIELDS = ("run_id", "variant", "queries", "stage", "mean_ms", "fraction_of_staged_forward",
          "round_mean_min_ms", "round_mean_max_ms", "samples", "bare_whole_ms", "staged_whole_ms",
          "instrumentation_overhead_percent")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stats(values):
    require(bool(values) and all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in values),
            "Stage timing samples must be finite positive numbers")
    return {"mean_ms": math.fsum(values) / len(values), "median_ms": statistics.median(values),
            "min_ms": min(values), "max_ms": max(values), "samples": len(values)}


def compare(actual, saved):
    require(actual.keys() == saved.keys(), "Stage summary fields differ")
    for key, value in actual.items():
        if isinstance(value, dict):
            compare(value, saved[key])
        else:
            require(type(saved[key]) in (float, int) and math.isclose(value, saved[key], rel_tol=1e-9, abs_tol=1e-9),
                    "Stage summary differs from raw samples: " + key)


def summarize(raw):
    """Recompute attribution from all randomized blocks, never trust summary alone."""
    require((raw["rounds"], raw["warmup_each_block"], raw["repetitions_each_block"]) == (5, 10, 50),
            "Unexpected stage timing protocol")
    require(set(raw["models"]) == {"c0", "c2"}, "Stage assessment must contain both paired models")
    result = {}
    expected = {(r, q, mode) for r in range(1, 6) for q in (300, 200, 150) for mode in ("bare", "staged")}
    for variant, model in raw["models"].items():
        require(model["model_state_unchanged"] is True and model["query_budget_restored"] is True
                and model["hooks_remaining"] == 0 and model["identity_checks"] == {str(q): True for q in (300, 200, 150)},
                "Stage timing identity/restoration gate failed")
        blocks = model["blocks"]
        require(len(blocks) == len(expected) and {(b["round"], b["queries"], b["mode"]) for b in blocks} == expected,
                "Missing or duplicate stage timing block")
        for block in blocks:
            values = block["measurements"]
            require(set(values) == ({"whole", *STAGES} if block["mode"] == "staged" else {"whole"}),
                    "Unexpected measurement stages")
            for samples in values.values():
                require(len(samples) == 50, "Incomplete stage timing block")
                stats(samples)
            if block["mode"] == "staged":
                require(all(abs(math.fsum(values[s][i] for s in STAGES) - total) <= .02
                            for i, total in enumerate(values["whole"])), "Stage sum does not match whole forward")
        summary = {}
        for budget in (300, 200, 150):
            selected = [b for b in blocks if b["queries"] == budget]
            bare = stats([v for b in selected if b["mode"] == "bare" for v in b["measurements"]["whole"]])
            staged = [b for b in selected if b["mode"] == "staged"]
            whole = stats([v for b in staged for v in b["measurements"]["whole"]])
            item = {"bare_whole": bare, "staged_whole": whole,
                    "instrumentation_overhead_percent": 100 * (whole["mean_ms"] / bare["mean_ms"] - 1), "stages": {}}
            for stage in STAGES:
                entry = stats([v for b in staged for v in b["measurements"][stage]])
                means = [stats(b["measurements"][stage])["mean_ms"] for b in staged]
                entry.update(fraction_of_staged_forward=entry["mean_ms"] / whole["mean_ms"],
                             round_mean_min_ms=min(means), round_mean_max_ms=max(means))
                item["stages"][stage] = entry
            summary[str(budget)] = item
        compare(summary, model["summary"])
        result[variant] = {"summary": summary}
    return result


def load_archive(root=ROOT):
    root = Path(root)
    path, identity_path = root / EVIDENCE, root / IDENTITY
    raw = json.loads(path.read_text())
    identity = json.loads(identity_path.read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    require(identity["schema"] == "5090.stage-latency-archive.v1", "Unknown stage identity schema")
    require(digest(path) == identity["assessment_sha256"], "Stage assessment SHA mismatch")
    source = (root / identity["executed_source"]).resolve()
    require(source.is_relative_to(root.resolve()) and digest(source) == identity["executed_source_sha256"],
            "Executed profiling source SHA mismatch")
    models = summarize(raw)
    for variant, model in models.items():
        checkpoint = identity["checkpoints"][variant]
        require(checkpoint["path"] == raw["models"][variant]["checkpoint"], "Checkpoint path mismatch")
        model["checkpoint_sha256"] = checkpoint["sha256"]
    return {"scope": raw["scope"], "measurement_started_at": identity["measurement_started_at"],
            "evidence": EVIDENCE, "identity": IDENTITY, "assessment_sha256": identity["assessment_sha256"],
            "environment": raw["environment"], "models": models}


def collect(records):
    """Attach this fixed assessment only to the exact completed training weights."""
    result = load_archive()
    matches = {}
    for variant, model in result["models"].items():
        matches[variant] = [r for r in records if r["variant"] == variant
                            and r["role"] == "paired_150e_screen" and r["training_status"] == "completed"
                            and r["checkpoint_sha256"].get("best") == model["checkpoint_sha256"]]
    if not any(matches.values()):
        return None
    require(all(len(items) == 1 for items in matches.values()), "Stage assessment training identity is missing or ambiguous")
    for variant, items in matches.items():
        result["models"][variant]["run_id"] = items[0]["run_id"]
    return result


def rows(data):
    assessment = data.get("stage_assessment")
    if assessment:
        for variant, model in sorted(assessment["models"].items()):
            for budget in (300, 200, 150):
                item = model["summary"][str(budget)]
                for stage in STAGES:
                    yield {"run_id": model["run_id"], "variant": variant, "queries": budget, "stage": stage,
                           **{key: item["stages"][stage][key] for key in FIELDS[4:9]},
                           "bare_whole_ms": item["bare_whole"]["mean_ms"],
                           "staged_whole_ms": item["staged_whole"]["mean_ms"],
                           "instrumentation_overhead_percent": item["instrumentation_overhead_percent"]}


def markdown(data):
    assessment = data.get("stage_assessment")
    if not assessment:
        return []
    models = assessment["models"]
    lines = ["", "## 5090 分段延迟与瓶颈评估", "",
             "固定一张真实验证图；未融合FP32、batch1、640，先预热150次，再做5轮交错顺序，每条件每轮10次预热/50次计时。分段与无标记原生forward分别采样；每条件每模式250次。此处为独立测量，不替换上方200次整网计时。",
             "同一次forward内设置CUDA Event边界，分段之和逐次核对整网；包含CPU提交导致的CUDA流空隙，不等于纯GPU内核忙碌时间，不含传输/预处理/后处理。", "",
             "| 阶段（300Q） | C0 ms | C0占比 | C2 ms | C2占比 |", "|---|---:|---:|---:|---:|"]
    for stage, label in zip(STAGES, LABELS):
        cells = [label]
        for variant in ("c0", "c2"):
            entry = models[variant]["summary"]["300"]["stages"][stage]
            cells += [f"{entry['mean_ms']:.3f}", f"{100 * entry['fraction_of_staged_forward']:.1f}%"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "| 模型 | Q | 无标记整网 ms | 分段整网 ms | 标记开销 | Decoder与预测头 ms |",
              "|---|---:|---:|---:|---:|---:|"]
    for variant in ("c0", "c2"):
        for budget in (300, 200, 150):
            item = models[variant]["summary"][str(budget)]
            lines.append(f"| {variant.upper()} | {budget} | {item['bare_whole']['mean_ms']:.3f} | {item['staged_whole']['mean_ms']:.3f} | {item['instrumentation_overhead_percent']:.2f}% | {item['stages']['decoder_and_prediction_heads']['mean_ms']:.3f} |")
    lines += ["", "Backbone为最大单块（约39%），Neck/Encoder约35%，两者合计约74%；Decoder与预测头约20%。Query从300降到150时，Decoder段也未明显变快，不能用Query减半推算整网计算量或延迟减半。若仅假设Decoder段耗时减半、其余不变，整网耗时约下降10%，这只是占比估算。",
              "本实验只改变Query；Decoder仍为3层、8头、3层级、各层级4个采样点，未减少采样点或实施采样mask。固定特征图上的Backbone、Neck/Encoder和value投影等工作不随Query缩减同比下降。",
              "模块分布尚不能判定算力、访存或CPU提交哪个是根因；需要算子级分析及目标部署路径复测。固定单图结果不代表全验证集时延分布；FP16、融合、TensorRT或Orin需重新测量。输出/RNG等价、模型state不变、预算恢复和0残留hook均通过；checkpoint与输入哈希为归档时回溯核验，详见身份记录。", "",
              "[分段CSV](stage_latency.csv) · Excel的Stage latency页 · "
              f"[逐次原始计时](../../{assessment['evidence']}) · [身份与复现来源](../../{assessment['identity']}) · "
              "[完整方法及复现](../../docs/experiment_reports/2026-10-24_5090_baselines/README.md#5090分段延迟与瓶颈评估)", ""]
    return lines
