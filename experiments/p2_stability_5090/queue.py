"""Stage-resumable C0/C2 seed1/2 queue; completed arms are never retrained."""
import argparse
import fcntl
import json
from pathlib import Path
import statistics
import subprocess
import sys

from experiments.decoder_budget_5090.train_5090 import now, sha256_file, write_atomic
from experiments.p2_p5_la_5090.model import MODELS, build_model
from experiments.p2_p5_la_5090.train import source_hashes
from experiments.p2_stability_5090.same_gt import DATA, OLD, ROOT, read, table

DEFAULT_ROOT = ROOT / "runs/p2_stability_5090/5090-p2-stability-20261027-r1"
DATA_YAML = ROOT / "runs/p2_p5_la_5090/5090-la-only-150e-20261027-s0-r1/data.yaml"
ANNOTATIONS = DATA / "annotations/VisDrone2019-DET_val_coco.json"


def sources():
    return {**source_hashes(), **{p.relative_to(ROOT).as_posix(): sha256_file(p)
            for p in Path(__file__).parent.glob("*.py")}}


def training_command(root, arm, seed, smoke=False):
    return [sys.executable, "-u", "-m", "experiments.p2_p5_la_5090.train", "--variant", arm,
            "--data", str(root / "data.yaml"), "--project", str(root / ("smoke" if smoke else "training")),
            "--name", f"5090-{arm}-{'smoke' if smoke else '150e'}-s{seed}",
            "--epochs", "1" if smoke else "150", "--seed", str(seed), "--fraction", ".01" if smoke else "1"]


def validate_training(record, arm, seed):
    if (record["status"] != "completed" or record["variant"] != arm or record["seed"] != seed
            or record["completed_epochs"] != 150 or record["fraction"] != 1
            or record["recipe"]["pretrained"] or record["recipe"]["resume"]):
        raise ValueError("Require completed scratch 150e receipt for this arm/seed")
    for item in record["checkpoints"].values():
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError("Training checkpoint identity mismatch")


def paired_summary(root):
    rows = []
    for seed in (0, 1, 2):
        records = {}
        for arm in ("c0", "c2"):
            if seed == 0:
                d = read(OLD / f"5090-rtdetr-r18-{arm}-150e-s0-query-full/summary.json")
                records[arm] = next(v for v in d["conditions"] if v["queries"] == 300)["metrics"]
            else:
                p = root / f"evaluations/{arm}-s{seed}/summary.json"
                if p.exists():
                    records[arm] = read(p)["metrics"]
        if len(records) == 2:
            rows.append({"seed": seed, **{f"{arm}_{key}": 100 * records[arm][key]
                         for arm in ("c0", "c2") for key in ("AP", "APs", "AP50", "AP75")},
                         **{f"delta_{key}_pp": 100 * (records["c2"][key] - records["c0"][key])
                         for key in ("AP", "APs", "AP50", "AP75")}})
    table(root / "paired_seeds.csv", rows)
    deltas = [r["delta_AP_pp"] for r in rows]
    result = {"completed_pairs": len(rows), "rows": rows, "mean_delta_AP_pp": statistics.mean(deltas),
              "min_delta_AP_pp": min(deltas), "max_delta_AP_pp": max(deltas),
              "sample_std_delta_AP_pp": statistics.stdev(deltas) if len(deltas) > 1 else None,
              "engineering_gate": "three positive paired AP differences and mean >=0.3pp; not a significance test",
              "gate_passed": len(rows) == 3 and min(deltas) > 0 and statistics.mean(deltas) >= .3,
              "small_target_claim": "inspect APs deltas separately; overall AP alone cannot establish a small-target benefit"}
    write_atomic(root / "paired_summary.json", result)


def run(args):
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / "logs").mkdir(exist_ok=True)
        state_path = root / "status.json"
        state = read(state_path) if state_path.exists() else {"status": "preparing", "stages": [], "created_utc": now()}

        def save():
            state["updated_utc"] = now()
            write_atomic(state_path, state)

        pin_path = root / "plan.json"
        if pin_path.exists():
            pin = read(pin_path)
            if (pin["source_sha256"] != sources() or pin["annotations_sha256"] != sha256_file(ANNOTATIONS)
                    or pin["data_yaml_sha256"] != sha256_file(root / "data.yaml")):
                raise RuntimeError("Frozen source/data changed; inspect before continuing this experiment")
        else:
            (root / "data.yaml").write_bytes(DATA_YAML.read_bytes())
            pin = {"seeds": [1, 2], "variants": ["c0", "c2"], "epochs": 150, "from_scratch": True,
                   "source_sha256": sources(), "annotations_sha256": sha256_file(ANNOTATIONS),
                   "data_yaml_sha256": sha256_file(root / "data.yaml"), "created_utc": now(),
                   "resume_policy": "completed stages skipped; interrupted training preserved and requires explicit recovery audit; never warm-start best.pt"}
            write_atomic(pin_path, pin)

        def stage(name, command):
            previous = [s for s in state["stages"] if s["name"] == name]
            if previous and previous[-1]["status"] == "completed":
                return
            item = {"name": name, "command": command, "status": "running", "started_utc": now(),
                    "log": str(root / "logs" / f"{name}-attempt{len(previous) + 1}.log")}
            state.update(status="running", phase=name)
            state["stages"].append(item)
            save()
            with Path(item["log"]).open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state["child_pid"] = child.pid
                save()
                try:
                    while True:
                        try:
                            code = child.wait(timeout=15)
                            break
                        except subprocess.TimeoutExpired:
                            state["training_progress"] = {}
                            for p in (root / "training").glob("*/experiment.json"):
                                d = read(p)
                                state["training_progress"][p.parent.name] = {k: d[k] for k in
                                    ("status", "completed_epochs", "latest_internal_map", "best_internal_map") if k in d}
                            save()
                except BaseException:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise
            state["child_pid"] = None
            item.update(returncode=code, status="completed" if code == 0 else "failed", finished_utc=now())
            save()
            if code:
                raise RuntimeError(f"{name} failed with {code}: {item['log']}")
            if sources() != pin["source_sha256"] or sha256_file(ANNOTATIONS) != pin["annotations_sha256"]:
                raise RuntimeError("Source/GT identity changed during a stage")

        try:
            if not (root / "asset_audit.json").exists():
                stage("asset-audit", [sys.executable, str(ROOT / "docs/experiment_reports/2026-10-24_5090_baselines/audit_assets_5090.py"),
                                      "--output", str(root / "asset_audit.json")])
            audit, old = read(root / "asset_audit.json"), read(ROOT / "docs/experiment_reports/2026-10-24_5090_baselines/raw/asset_audit_5090.json")
            if (audit["status"] != "passed" or audit["coco"]["sha256"] != pin["annotations_sha256"]
                    or any(audit[k] != old[k] for k in ("label_content_sha256", "image_inventory_sha256"))):
                raise ValueError("Dataset audit differs from the seed0 baseline")
            if not (root / "initializations.json").exists():
                initial = {}
                import torch
                torch.set_num_threads(4)
                for seed in (1, 2):
                    initial[str(seed)] = {}
                    for arm in ("c0", "c2"):
                        model = build_model(str(MODELS[arm]), seed=seed)
                        initial[str(seed)][arm] = model.ablation_initialization
                        del model
                    for key in ("shared_state_sha256", "p5_slot_sha256", "r0_full_state_sha256"):
                        if initial[str(seed)]["c0"][key] != initial[str(seed)]["c2"][key]:
                            raise ValueError("Pair initialization mismatch")
                write_atomic(root / "initializations.json", initial)
            if args.prepare_only:
                state.update(status="prepared", phase="awaiting_execution")
                save()
                return
            if args.smoke:
                for arm in ("c0", "c2"):
                    stage("smoke-" + arm, training_command(root, arm, 1, smoke=True))
                state.update(status="smoke_completed", phase="awaiting_full_training")
                save()
                return
            for seed in args.seeds:
                pair = []
                for arm in ("c0", "c2"):
                    training = root / f"training/5090-{arm}-150e-s{seed}"
                    receipt = training / "experiment.json"
                    if not receipt.exists():
                        if training.exists():
                            raise RuntimeError(f"Incomplete run preserved at {training}; audit recovery instead of overwriting")
                        stage(f"train-{arm}-s{seed}", training_command(root, arm, seed))
                    record = read(receipt)
                    validate_training(record, arm, seed)
                    if record["source_sha256"] != source_hashes() or record["data_yaml_sha256"] != pin["data_yaml_sha256"]:
                        raise ValueError("Training receipt has different frozen source/data")
                    expected = read(root / "initializations.json")[str(seed)][arm]
                    if any(record["initialization"][k] != expected[k] for k in
                           ("shared_state_sha256", "p5_slot_sha256", "p2_slot_sha256", "r0_full_state_sha256")):
                        raise ValueError("Actual trainer initialization differs from the preflight")
                    pair.append(record)
                    output = root / f"evaluations/{arm}-s{seed}"
                    if not (output / "summary.json").exists():
                        if output.exists():
                            # Preserve failed output; evaluate into a fresh attempt and publish only on success.
                            output.rename(output.with_name(output.name + "-failed-" + now().replace(":", "")))
                        stage(f"eval-{arm}-s{seed}", [sys.executable, "-u", "-m", "experiments.p2_p5_la_5090.evaluate",
                              "--variant", arm, "--run", str(training), "--annotations", str(ANNOTATIONS),
                              "--images", str(DATA / "images/val"), "--output", str(output)])
                    evaluation = read(output / "summary.json")
                    if (evaluation["status"] != "completed" or evaluation["scope"] != "full_val"
                            or evaluation["checkpoint"]["sha256"] != record["checkpoints"]["best"]["sha256"]):
                        raise ValueError("Evaluation receipt mismatch")
                    paired_summary(root)
                if pair[0]["recipe"] != pair[1]["recipe"]:
                    raise ValueError("Paired training recipes differ")
            paired_summary(root)
            state.update(status="completed" if read(root / "paired_summary.json")["completed_pairs"] == 3 else "seed1_completed_seed2_pending",
                         phase="finished_selected_seeds")
            save()
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            save()
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--seeds", type=int, choices=(1, 2), nargs="+", default=[1])
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--status", action="store_true")
    args = p.parse_args()
    if args.status:
        print(json.dumps(read(args.root / "status.json"), indent=2, ensure_ascii=False))
    else:
        run(args)
