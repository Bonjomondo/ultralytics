"""Verify the launch archive without CUDA, SDK, datasets or checkpoints."""
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def read(name):
    return json.loads((HERE/"raw"/name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for line in (HERE/"MANIFEST.sha256").read_text().splitlines():
        digest,name=line.split("  ",1);path=(ROOT/name).resolve()
        assert path.is_relative_to(ROOT) and sha(path)==digest,name
    frozen=read("frozen_sources.json")
    for name,digest in frozen["source_sha256"].items():
        assert sha(ROOT/name)==digest,name
    for name,digest in frozen["inputs"].items():
        base=Path(name).name
        if base in ("data.yaml","initialization_plan.json"):
            assert sha(HERE/"raw"/base)==digest,base
        else:
            assert base=="VisDrone2019-DET_val_coco.json"
            assert read("asset_audit.json")["coco"]["sha256"]==digest
    plan=read("initialization_plan.json");gates=read("gates_summary.json")
    assert not plan["checkpoint_initialization"] and gates["status"]=="passed"
    assert len(gates["variants"])==len(plan["variants"])==4
    for v in gates["variants"]:
        assert v["status"]=="completed" and v["epochs"]==1 and v["fraction"]==.01
        assert v["common_state_sha256"]==plan["common_state_sha256"] and v["full_initial_state_verified"]
        assert v["initialization"]["full_initial_state_sha256"]==plan["variants"][v["variant"]]["initialization"]["full_initial_state_sha256"]
        assert v["initialization"]["pretrained"] is False and v["initialization"]["source_checkpoint"] is None
        assert v["reload_gate"]["images"]==8 and v["reload_gate"]["repeated_forward_exact"] and v["reload_gate"]["model_state_unchanged"]
        assert v["health"]["p2_first_weight_grad_abs"]>0
        assert all(v["health"][f"d{i+1}_offset_weight_grad_abs"]>0 for i in range(v["architecture"]["decoder_layers"]))
        assert v["checkpoints"]["best"]["sha256"]==v["reload_gate"]["checkpoint_sha256"]
    training=read("first_training_status.json")
    assert training["epochs_requested"]==150 and training["fraction"]==1 and training["role"]=="fresh_150e"
    assert training["full_initial_state_verified"] and not training["pretrained"] and not training["resume"]
    assert read("validation.json")["passed"]==10
    for document in ("README.md","REPORT.md"):
        for target in re.findall(r"\]\(([^)]+)\)",(HERE/document).read_text()):
            if not target.startswith(("http:","https:")):
                assert (HERE/target.split("#",1)[0]).exists(),target
    print("Verified four fresh architectures/initial states, actual gradient+reload gates, full150 launch and cloud-only manifest/links")


if __name__=="__main__":main()
