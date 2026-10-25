"""Check the committed 5090 launch evidence without server data, weights or CUDA."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LAUNCH_COMMIT = "d11de6e4b8b48e17eedce315f1d56d6ecb867f87"
HISTORICAL_INPUTS = {"experiments/decoder_budget_5090/queue_5090.py",
                     "experiments/decoder_budget_5090/tests/test_queue_5090.py"}


def read(name):
    return json.loads((HERE / "raw" / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matches_source(path, expected):
    if path.is_file() and sha(path) == expected:
        return True
    relative = path.relative_to(ROOT).as_posix()
    if relative not in HISTORICAL_INPUTS:
        return False
    # The reporter extends the queue after the completed experiment. Validate
    # the actual launch bytes in Git instead of rewriting its frozen SHA map.
    original = subprocess.check_output(["git", "show", f"{LAUNCH_COMMIT}:{relative}"], cwd=ROOT)
    return hashlib.sha256(original).hexdigest() == expected


def main():
    manifest = (HERE / "MANIFEST.sha256").read_text().splitlines()
    for line in manifest:
        expected, relative = line.split("  ", 1)
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT) and matches_source(path, expected), relative
    sys.path.insert(0, str(ROOT))
    from experiments.decoder_budget_5090.stage_latency_5090 import load_archive
    stages = load_archive(ROOT)
    assert set(stages["models"]) == {"c0", "c2"}
    audit = read("asset_audit_5090.json")
    assert audit["status"] == "passed" and not audit["errors"]
    assert audit["splits"]["train"]["images"] == 6471
    assert audit["splits"]["val"]["images"] == 548
    assert sha(HERE / "audit_assets_5090.py") == audit["audit_script_sha256"]
    gates = read("gates_passed_5090.json")
    assert gates["status"] == "passed"
    assert gates["training"]["c0"]["shared_state_sha256"] == gates["training"]["c2"]["shared_state_sha256"]
    assert sha(HERE / "raw/source_pin_5090.json") == gates["source_pin_sha256"]
    training, sweeps = read("training_gates_5090.json"), read("query_gates_5090.json")
    for variant in ("c0", "c2"):
        assert training[variant]["status"] == "completed" and training[variant]["completed_epochs"] == 1
        assert training[variant]["fraction"] == .01
        assert "5090" in training[variant]["environment"]["gpu"]
        assert training[variant]["initialization"]["shared_state_verified"]
        assert [r["queries"] for r in sweeps[variant]["conditions"]] == [300, 250, 200, 150]
        assert all(r["images"] == 8 for r in sweeps[variant]["conditions"])
        assert len({r["input_gt_identity_sha256"] for r in sweeps[variant]["conditions"]}) == 1
        assert all(r["all_batches_passed"] for r in sweeps[variant]["native_observation_checks"].values())
    assert training["c2"]["latest_epoch_health"]["p2_first_weight_grad_abs"] > 0
    pin_root = Path(read("source_pin_5090.json")["repo_root"])
    for source, expected in read("source_sha256_5090.json").items():
        path = Path(source)
        if path.is_relative_to(pin_root):
            relative = path.relative_to(pin_root)
            local = HERE / "raw/data_5090.yaml" if relative.name == "data_5090.yaml" else ROOT / relative
            assert matches_source(local, expected), relative
    assert re.search(r"\b58 passed\b", (HERE / "raw/pytest_5090.log").read_text())
    for target in re.findall(r"\]\(([^)]+)\)", (HERE / "README.md").read_text()):
        target = target.split("#", 1)[0]
        if target and not target.startswith(("http:", "https:")):
            assert (HERE / target).exists(), target
    print(f"Verified {len(manifest)} files, stage timing evidence and the real-data 5090 gates; full150 status is an archived observation")


if __name__ == "__main__":
    main()
