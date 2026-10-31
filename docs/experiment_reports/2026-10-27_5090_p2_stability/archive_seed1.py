"""Audit finished seed1 assets and export compact evidence; no model execution.

Run from the repository root with the verified training venv and PYTHONPATH.
The destination must be new. Training source, predictions and weights are read only.
"""
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
import shutil
import statistics
import sys
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[3]
RUN = REPO / "runs/p2_stability_5090/5090-p2-stability-20261027-r1"
OLD = REPO / "runs/decoder_budget_5090/5090-c0c2-baselines-20261024-s0-r1"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def archive(output):
    import detrscope
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import ultralytics

    assert Path(ultralytics.__file__).resolve().is_relative_to(REPO / "ultralytics")
    assert Path(detrscope.__file__).resolve().is_relative_to(REPO.parent / "detrscope/src")
    pin, status = read(RUN / "plan.json"), read(RUN / "status.json")
    assert status["status"] == "seed1_completed_seed2_pending"
    assert all(s["status"] == "completed" and s["returncode"] == 0 for s in status["stages"])
    assert all(sha(REPO / p) == digest for p, digest in pin["source_sha256"].items())
    assert sha(RUN / "data.yaml") == pin["data_yaml_sha256"]
    assert sha(REPO.parent / "SO_DETR/datasets/VisDrone2019-DET-YOLO/annotations/VisDrone2019-DET_val_coco.json") == pin["annotations_sha256"]
    initials = read(RUN / "initializations.json")["1"]
    evaluations, trainings, curves, health, identities = {}, {}, [], [], []
    for arm in ("c0", "c2"):
        train = RUN / f"training/5090-{arm}-150e-s1"
        evaluation = RUN / f"evaluations/{arm}-s1/summary.json"
        t, e = read(train / "experiment.json"), read(evaluation)
        trainings[arm], evaluations[arm] = t, e
        assert t["status"] == e["status"] == "completed"
        assert t["seed"] == e["seed"] == 1 and t["completed_epochs"] == 150
        assert t["fraction"] == 1 and not t["recipe"]["resume"] and not t["recipe"]["pretrained"]
        assert e["scope"] == "full_val" and len(e["protocol"]["image_ids"]) == 548
        assert e["protocol"]["formal_max_dets"] == [1, 10, 100]
        assert e["training_receipt_sha256"] == sha(train / "experiment.json")
        assert e["source_sha256"] == e["training_source_sha256"] == t["source_sha256"]
        assert all(pin["source_sha256"][p] == v for p, v in t["source_sha256"].items())
        assert e["checkpoint"] == t["checkpoints"]["best"] and e["model_state_unchanged"]
        assert sha(evaluation.with_name("predictions.json")) == e["predictions_sha256"]
        for item in t["checkpoints"].values():
            path = Path(item["path"])
            assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
        assert all(t["initialization"][k] == initials[arm][k] for k in
                   ("shared_state_sha256", "p5_slot_sha256", "r0_full_state_sha256", "p2_slot_sha256"))
        with (train / "results.csv").open() as stream:
            rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(stream)]
        hs = [json.loads(line) for line in (train / "health.jsonl").read_text().splitlines()]
        assert [int(r["epoch"]) for r in rows] == [h["epoch"] for h in hs] == list(range(1, 151))
        assert all(math.isfinite(float(v)) for r in rows for v in r.values())
        assert all(math.isfinite(v) for h in hs for v in h.values() if isinstance(v, (int, float)))
        assert all(v == 1 for h in hs for k, v in h.items() if k.endswith("_finite"))
        if arm == "c2":
            assert min(h["p2_first_weight_grad_abs"] for h in hs) > 0
        curves.extend({"variant": arm, **r} for r in rows)
        health.extend({"variant": arm, **h} for h in hs)
        best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
        identities.append({"variant": arm, "epochs": len(rows), "health_epochs": len(hs),
                           "best_internal_epoch": int(best["epoch"]),
                           "best_internal_map_percent": 100 * float(best["metrics/mAP50-95(B)"]),
                           "checkpoint_hashes_verified": True, "predictions_hash_verified": True,
                           "curve_and_sampled_health_finite": True,
                           "p2_sampled_gradient_min": min(h["p2_first_weight_grad_abs"] for h in hs) if arm == "c2" else None,
                           "peak_allocated_mib": max(h["train_peak_allocated_mib"] for h in hs)})
    assert trainings["c0"]["recipe"] == trainings["c2"]["recipe"]
    assert evaluations["c0"]["protocol"] == evaluations["c2"]["protocol"]
    for k in ("shared_state_sha256", "p5_slot_sha256", "r0_full_state_sha256"):
        assert initials["c0"][k] == initials["c2"][k]

    rows, categories, original_seed0 = [], [], {}
    previous_categories = REPO / "docs/experiment_reports/2026-10-27_5090_la_only_150e_results/category_metrics.csv"
    prior = list(csv.DictReader(previous_categories.open()))
    for seed in (0, 1):
        models = {}
        for arm in ("c0", "c2"):
            if seed == 0:
                path = OLD / f"5090-rtdetr-r18-{arm}-150e-s0-query-full/summary.json"
                old = read(path)
                condition = next(c for c in old["conditions"] if c["queries"] == 300)
                assert condition["images"] == 548
                assert old["protocol"]["annotation_sha256"] == evaluations[arm]["protocol"]["annotations_sha256"]
                models[arm] = condition["metrics"]
                original_seed0[arm] = {"source": str(path), "sha256": sha(path),
                                       "protocol": old["protocol"], "condition_q300": condition}
                categories.extend({"seed": seed, "variant": arm, "category": r["category"],
                                   **{k: 100 * float(r["native_" + k]) for k in ("AP", "APs", "APm", "APl")}}
                                  for r in prior if r["variant"] == arm)
            else:
                models[arm] = evaluations[arm]["metrics"]
                categories.extend({"seed": seed, "variant": arm, "category": name,
                                   **{k: 100 * v if v is not None else None for k, v in metrics.items()}}
                                  for name, metrics in evaluations[arm]["per_class"].items())
            rows.append({"seed": seed, "variant": arm, **{k: 100 * v for k, v in models[arm].items()}})
    stats = []
    for key in evaluations["c0"]["metrics"]:
        ds = [next(r[key] for r in rows if r["seed"] == s and r["variant"] == "c2") -
              next(r[key] for r in rows if r["seed"] == s and r["variant"] == "c0") for s in (0, 1)]
        stats.append({"metric": key, "seed0_delta_pp": ds[0], "seed1_delta_pp": ds[1],
                      "mean_delta_pp": statistics.mean(ds), "sample_std_delta_pp": statistics.stdev(ds),
                      "min_delta_pp": min(ds), "max_delta_pp": max(ds)})
    output.mkdir(parents=True, exist_ok=False)
    for arm in ("c0", "c2"):
        shutil.copyfile(RUN / f"training/5090-{arm}-150e-s1/experiment.json", output / f"training-{arm}.json")
        shutil.copyfile(RUN / f"evaluations/{arm}-s1/summary.json", output / f"evaluation-{arm}.json")
    for name in ("paired_seeds.csv", "paired_summary.json", "status.json"):
        shutil.copyfile(RUN / name, output / name)
    for name, data in (("formal_metrics.csv", rows), ("category_metrics.csv", categories),
                       ("paired_statistics.csv", stats), ("training_curves.csv", curves), ("training_health.csv", health)):
        table(output / name, data)
    summary = {"generated_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "server_run": str(RUN), "audit": identities, "frozen_source_files_verified": len(pin["source_sha256"]),
               "analysis_environment": {"executable": sys.executable, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
                                        "python": sys.version, "torch": torch.__version__, "matplotlib": matplotlib.__version__,
                                        "ultralytics_file": ultralytics.__file__, "detrscope_file": detrscope.__file__},
               "historical_environment": "training/evaluation receipts record the environments at execution time",
               "seed0_sources": original_seed0,
               "seed0_categories_source": {"path": str(previous_categories), "sha256": sha(previous_categories)},
               "statistics": stats, "gate_passed": False,
               "gate_reason": "seed2 pending; seed1 AP difference is negative, violating the predeclared all-three-positive criterion",
               "health_scope": "bbox/gradient/activation probes sample the first three training batches of each epoch; curves cover all epochs"}
    (output / "audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for arm, color in (("c0", "#41678d"), ("c2", "#d88034")):
        rs = [r for r in curves if r["variant"] == arm]
        axes[0].plot([int(r["epoch"]) for r in rs], [100 * float(r["metrics/mAP50-95(B)"]) for r in rs], label=arm.upper(), color=color)
    axes[0].set(xlabel="Epoch", ylabel="Internal validation mAP (%)", title="Seed1 training (internal metric)")
    axes[0].legend()
    ap = next(s for s in stats if s["metric"] == "AP")
    aps = next(s for s in stats if s["metric"] == "APs")
    for offset, record, color in ((-.17, ap, "#41678d"), (.17, aps, "#d88034")):
        ys = [record[f"seed{s}_delta_pp"] for s in (0, 1)]
        bars = axes[1].bar([s + offset for s in (0, 1)], ys, width=.32, label=record["metric"], color=color)
        axes[1].bar_label(bars, fmt="%+.3f", padding=3, fontsize=9)
    axes[1].axhline(0, color="#333333", linewidth=.8)
    axes[1].set(xticks=[0, 1], xticklabels=["Seed0", "Seed1"], ylabel="C2 - C0 (percentage points)", title="Full-val formal COCO differences", ylim=(-.22, 1.05))
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(output / "summary.svg")
    plt.close(fig)
    print("Verified source/data, paired initialization, completed receipts, checkpoint/prediction hashes and 300 epochs; wrote", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "seed1_results")
    archive(parser.parse_args().output)
