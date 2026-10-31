"""Verify this compact archive without CUDA, datasets or server checkpoints."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
for line in (root / "MANIFEST.sha256").read_text().splitlines():
    expected, relative = line.split("  ", 1)
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
summary = json.loads((root / "same_gt/summary.json").read_text())
assert summary["status"] == "completed" and summary["images"] == 548 and summary["valid_gt"] == 38759
with (root / "same_gt/groups.csv").open() as f:
    for row in csv.DictReader(f):
        n = sum(int(row[k]) for k in ("both", "neither", "recovered", "lost"))
        assert n == int(row["gt_count"])
        assert int(row["net_recovered"]) == int(row["recovered"]) - int(row["lost"])
deployment = json.loads((root / "deployment/summary.json").read_text())
assert deployment["status"] == "completed" and set(deployment["models"]) == {"d2", "d3"}
assert deployment["protocol"]["formal_max_dets"] == [1, 10, 100]
assert len(deployment["protocol"]["image_ids"]) == 548
assert len({v["checkpoint"]["sha256"] for v in deployment["models"].values()}) == 1
for seed in ("1", "2"):
    pair = json.loads((root / "raw/initializations.json").read_text())[seed]
    assert pair["c0"]["shared_state_sha256"] == pair["c2"]["shared_state_sha256"]
final = root / "seed1_results"
state = json.loads((final / "status.json").read_text())
assert state["status"] == "seed1_completed_seed2_pending"
assert all(s["status"] == "completed" and s["returncode"] == 0 for s in state["stages"])
with (final / "formal_metrics.csv").open() as f:
    metrics = {(int(r["seed"]), r["variant"]): r for r in csv.DictReader(f)}
assert set(metrics) == {(s, a) for s in (0, 1) for a in ("c0", "c2")}
for arm in ("c0", "c2"):
    training_path = final / f"training-{arm}.json"
    t = json.loads(training_path.read_text())
    e = json.loads((final / f"evaluation-{arm}.json").read_text())
    assert t["status"] == e["status"] == "completed" and t["completed_epochs"] == 150
    assert e["training_receipt_sha256"] == hashlib.sha256(training_path.read_bytes()).hexdigest()
    assert e["checkpoint"] == t["checkpoints"]["best"]
    assert len(e["protocol"]["image_ids"]) == 548 and e["protocol"]["formal_max_dets"] == [1, 10, 100]
    for k, v in e["metrics"].items():
        assert math.isclose(100 * v, float(metrics[(1, arm)][k]), abs_tol=1e-12)
for name in ("training_curves.csv", "training_health.csv"):
    with (final / name).open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 300
    for arm in ("c0", "c2"):
        assert [int(r["epoch"]) for r in rows if r["variant"] == arm] == list(range(1, 151))
    assert all(math.isfinite(float(v)) for r in rows for k, v in r.items() if k != "variant" and v)
with (final / "paired_statistics.csv").open() as f:
    for r in csv.DictReader(f):
        key = r["metric"]
        deltas = [float(metrics[(s, "c2")][key]) - float(metrics[(s, "c0")][key]) for s in (0, 1)]
        assert math.isclose(float(r["mean_delta_pp"]), statistics.mean(deltas), abs_tol=1e-12)
        assert math.isclose(float(r["sample_std_delta_pp"]), statistics.stdev(deltas), abs_tol=1e-12)
assert float(metrics[(1, "c2")]["AP"]) < float(metrics[(1, "c0")]["AP"])
print("Archive hashes, GT/deployment identity, paired initialization, final receipts, 300 curve/health epochs and paired statistics passed.")
