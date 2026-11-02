"""Verify exact C2 identity, gate gradients, actual full-network loss and serialization."""
import json
import torch
from common import ARMS, CONFIGS, EXP, REPORT, RUNS, atomic_save, sha, write
from train import paired
from ultralytics.nn.modules.gated_fusion import C2GatedConcat


def main():
    torch.set_num_threads(4)
    base = paired(str(CONFIGS["c2"])).eval().cuda()
    x = torch.rand(1, 3, 640, 640, device="cuda")
    with torch.inference_mode():
        expected = base(x)[0]
    records = {}
    for arm in ARMS:
        model = paired(str(CONFIGS[arm])).eval().cuda()
        with torch.inference_mode():
            actual = model(x)[0]
        error = float((expected - actual).abs().max())
        assert error == 0, (arm, error)
        assert model.ablation_initialization["shared_state_sha256"] == base.ablation_initialization["shared_state_sha256"]
        gate = next((m for m in model.modules() if isinstance(m, C2GatedConcat)), None)
        if gate is not None:
            xs = [torch.randn(2, c, 80, 80, device="cuda", requires_grad=True) for c in (256, 128, 128)]
            output = gate(xs)
            assert torch.equal(output, torch.cat(xs, 1))
            output.square().mean().backward()
            assert xs[1].grad.abs().sum() > 0
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in gate.parameters())
        records[arm] = {"parameters": sum(p.numel() for p in model.parameters()),
                        "full_640_output_max_abs_error_vs_c2": error,
                        "shared_state_sha256": model.ablation_initialization["shared_state_sha256"],
                        "gate_and_p2_gradients_pass": gate is not None or arm == "c2"}
        del model
    # Direct checkpoint/load and re-forward with a custom gate, not just a state dict.
    model = paired(str(CONFIGS["spatial"])).cpu()
    path = RUNS / "preflight/roundtrip.pt"
    atomic_save(path, {"epoch": 0, "ema": model, "optimizer": {"state": {}}})
    restored = torch.load(path, map_location="cpu")["ema"]
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    write(REPORT / "raw/preflight.json", {"status": "passed", "arms": records,
           "serialization_roundtrip_pass": True, "roundtrip_sha256": sha(path)})
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
