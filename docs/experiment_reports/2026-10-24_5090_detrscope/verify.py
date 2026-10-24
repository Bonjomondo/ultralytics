"""Verify the committed phase-0 evidence using Python's standard library only."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE / "raw" / name).read_text())


def main():
    entries = {}
    for line in (HERE / "MANIFEST.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT), relative
        assert sha(path) == expected, relative
        entries[relative] = expected
    sources = read("experiment_sources.json")
    for relative, expected in sources.items():
        assert sha(ROOT / relative) == expected, relative
    for name in ("cpu_smoke.json", "gpu_smoke.json"):
        smoke = read(name)
        assert smoke["status"] == "passed"
        assert smoke["model_state_unchanged"] and smoke["query_budget_restored"]
        assert smoke["hooks_remaining"] == 0
        assert [row["queries"] for row in smoke["conditions"]] == [300, 250, 200, 150]
        assert all(row["identity_passed_all"] and row["images"] == 2 and row["executed_layers"] == [1, 2, 3]
                   for row in smoke["conditions"])
        for name, expected in smoke["experiment_sources"].items():
            assert sha(ROOT / "experiments/decoder_budget_5090" / name) == expected, name
    preflight = read("preflight.json")
    assert preflight["status"] == "passed" and not preflight["errors"]
    assert "5090" in preflight["environment"]["gpu"]
    assert preflight["environment"]["cuda_available"]
    validation = read("validation.json")
    for device in ("cpu", "cuda"):
        log = (HERE / "raw" / f"pytest_{device}.log").read_text()
        assert re.search(rf"\b{validation[device + '_tests']['passed']} passed\b", log), device
    assert sha(HERE / "RESEARCH_PLAN.md") == read("asset_audit.json")["original_research_plan_sha256"]
    documents = [*HERE.glob("*.md"), ROOT / "experiments/decoder_budget_5090/README.md"]
    for document in documents:
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            target = target.split("#", 1)[0]
            if target and not target.startswith(("http:", "https:")):
                assert (document.parent / target).exists(), (document, target)
    print(f"Verified {len(entries)} committed files, source hashes, CPU/CUDA evidence and report links")


if __name__ == "__main__":
    main()
