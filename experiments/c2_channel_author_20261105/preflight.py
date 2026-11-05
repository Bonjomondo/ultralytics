"""Identity, pairing, gradient and initial checkpoint checks for all three arms."""
import json
import torch
from common import ARMS, CONFIGS, REPORT, RUNS, sha, write
from train import paired
from ultralytics.nn.modules.p2_channel_gate import P2GateConcat


def main():
    torch.set_num_threads(8)
    results = {}
    # Archive every complete fresh initialization for later verification, not just a seed replay.
    for seed in [0, 1, 2]:
        reference = paired(str(CONFIGS["c2"]), seed).eval().cuda()
        x = torch.rand(1, 3, 640, 640, device="cuda")
        with torch.inference_mode():expected = reference(x)[0]
        initial = RUNS / f"initialization_s{seed}.pt"
        initial.parent.mkdir(parents=True, exist_ok=True)
        torch.save(reference.cpu().state_dict(), initial)
        shared = reference.initialization_receipt["shared_sha256"]
        for arm in ARMS:
            model = paired(str(CONFIGS[arm]), seed).eval().cuda()
            assert model.initialization_receipt["shared_sha256"] == shared
            with torch.inference_mode():error = float((expected - model(x)[0]).abs().max())
            assert error == 0
            if arm != "c2":
                gate = model.model[18]
                assert isinstance(gate, P2GateConcat)
                assert torch.equal(gate.alpha(), torch.ones_like(gate.alpha()))
                xs = [torch.rand(2, c, 8, 8, device="cuda", requires_grad=True) for c in [256, 128, 128]]
                gate(xs).square().mean().backward()
                assert torch.all(gate.logits.grad.abs() > 0) and xs[1].grad.abs().sum() > 0
            results[f"{arm}_s{seed}"] = {"parameters": sum(p.numel() for p in model.parameters()),
                "max_abs_error_vs_c2_640": error, "shared_sha256": shared, "gradient_test_pass": True}
            del model
        del reference
        results[f"initialization_s{seed}"] = {"path": str(initial), "sha256": sha(initial)}
    write(REPORT / "raw/preflight.json", {"status": "passed", "tests": results})
    print(json.dumps(results), flush=True)


if __name__ == "__main__":main()
