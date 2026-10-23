"""Train one paired P2-SFI arm using the existing R18 recipe, without pretrained weights."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from train_rtdetr_r18_na import RECIPE, TrainingHealth
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import unwrap_model

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ultralytics/cfg/models/rt-detr"
MODELS = {"c0": CONFIG / "rtdetr-resnet18.yaml", **{
    name: CONFIG / f"rtdetr-resnet18-p2-sfi-{name}.yaml" for name in ("c1", "c2", "c3")}}


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_model(cfg, nc=10, ch=3, seed=0, weights=None, verbose=False):
    """Copy by stable layer identity, excluding the replaced P3-projection/P2 slot."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        baseline = RTDETRDetectionModel(str(MODELS["c0"]), nc=nc, ch=ch, verbose=False)
        torch.manual_seed(seed + 1000)
        model = RTDETRDetectionModel(cfg, nc=nc, ch=ch, verbose=verbose)
        reference, initial = baseline.state_dict(), model.state_dict()
        is_baseline = model.yaml["head"] == baseline.yaml["head"]
        common = {name: value for name, value in reference.items()
                  if not name.startswith("model.14.")}
        for name, value in common.items():
            if name not in initial or value.shape != initial[name].shape:
                raise RuntimeError(f"Unexpected shared architecture mismatch: {name}")
        model.load_state_dict(reference if is_baseline else common, strict=is_baseline)
        actual = model.state_dict()
        if any(not torch.equal(actual[name], value) for name, value in common.items()):
            raise RuntimeError("Paired initialization failed")
        model.ablation_initialization = {
            "seed": seed, "new_branch_seed": seed + 1000,
            "source": "seeded R0 from scratch; no checkpoint or backbone pretrained weights",
            "r0_full_state_sha256": tensor_digest(reference),
            "shared_state_sha256": tensor_digest(common),
            "shared_state_items": len(common), "shared_state_verified": True,
            "excluded_slot": "model.14: baseline P3 projection versus candidate P2 downsampling",
            "variant_only_state": sorted(set(actual) - set(common)),
        }
        if weights is not None:
            if any(weights.yaml[key] != model.yaml[key] for key in ("backbone", "head")):
                raise ValueError("Resume checkpoint architecture differs from selected arm")
            model.load_state_dict(weights.float().state_dict(), strict=True)
            model.ablation_initialization = weights.ablation_initialization
    return model


class SFITrainer(RTDETRTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        return build_model(cfg, self.data["nc"], self.data["channels"], self.args.seed, weights, verbose)


class SFIHealth(TrainingHealth):
    """Sample encoder/D1-D3 matching queries, keeping denoising queries separate."""

    def start(self, trainer):
        if trainer.batch_size != 8 or trainer.args.amp:
            raise RuntimeError("The controlled recipe requires batch=8 and FP32")
        super().start(trainer)
        model = unwrap_model(trainer.model)
        head = model.model[-1]
        for name, mlp in [("encoder", head.enc_bbox_head), *[
            (f"d{i + 1}", layer) for i, layer in enumerate(head.dec_bbox_head)]]:
            def hidden(module, inputs, label=name):
                if module.training and self.capture:
                    value = inputs[0][:, -head.num_queries:].float()
                    self._sample(f"{label}_hidden_active_ratio", (value > 0).float().mean())
                    self._sample(f"{label}_hidden_query_variance", value.var(1, unbiased=False).mean())

            def delta(module, inputs, output, label=name):
                if module.training and self.capture:
                    value = output[:, -head.num_queries:].float()
                    self._sample(f"{label}_delta_query_variance", value.var(1, unbiased=False).mean())
                    self._sample(f"{label}_delta_abs", value.abs().mean())
                    self._sample(f"{label}_delta_finite", torch.isfinite(value).float().mean())

            self.handles.extend((mlp.layers[-1].register_forward_pre_hook(hidden),
                                 mlp.register_forward_hook(delta)))
        if model.model[14].f == 1:
            parameter = next(model.model[14].parameters())
            self.handles.append(parameter.register_hook(
                lambda grad: self._sample("p2_first_weight_grad_abs", grad.float().abs().mean())))

    def batch_end(self, trainer):
        if trainer.batch_size != 8:
            raise RuntimeError("Trainer changed the batch size; stop instead of changing this arm's recipe")
        if not torch.isfinite(trainer.loss).all():
            raise FloatingPointError("Nonfinite training loss; stopping this controlled queue")
        super().batch_end(trainer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=MODELS, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/detect/r18_p2_sfi_20261023")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = MODELS[args.variant]
    model = RTDETR(str(args.resume.resolve() if args.resume else cfg))
    if args.resume:
        expected = RTDETR(str(cfg)).model.yaml
        if any(model.model.yaml[key] != expected[key] for key in ("backbone", "head")):
            raise ValueError("Resume architecture mismatch")
    SFIHealth().attach(model)
    recipe = {**RECIPE, "epochs": args.epochs}
    model.train(trainer=SFITrainer, **recipe, data=str(ROOT / "configs/visdrone-local.yaml"),
                device="0", workers=0, fraction=args.fraction, project=str(args.project),
                name=args.name or f"rtdetr-r18-{args.variant}-150e-s0",
                pretrained=False, resume=str(args.resume.resolve()) if args.resume else False,
                exist_ok=bool(args.resume))


if __name__ == "__main__":
    main()
