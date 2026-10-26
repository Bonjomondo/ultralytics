"""Factorial P2/P5-LA experiment with independently paired module initialization."""
from pathlib import Path

import torch
import yaml

from experiments.r18_p2_sfi.continue_train import ContinuationTrainer
from experiments.r18_p2_sfi.train import SFIHealth, tensor_digest
from ultralytics.nn.modules import NAIFI
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import unwrap_model

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ultralytics/cfg/models/rt-detr"
MODELS = {
    "c0": CONFIG / "rtdetr-resnet18.yaml",
    "c2": CONFIG / "rtdetr-resnet18-p2-sfi-c2.yaml",
    "la": CONFIG / "rtdetr-resnet18-p5-naifi.yaml",
    "c2-la": CONFIG / "rtdetr-resnet18-p2-sfi-c2-p5-naifi.yaml",
}


def identify(cfg):
    config = yaml.safe_load(Path(cfg).read_text()) if not isinstance(cfg, dict) else cfg
    for name, path in MODELS.items():
        reference = yaml.safe_load(path.read_text())
        if all(config[key] == reference[key] for key in ("backbone", "head")):
            return name
    raise ValueError("Configuration is outside the fixed 300Q/4-point/3-layer factorial")


def inspect_model(model, variant):
    if identify(model.yaml) != variant:
        raise ValueError("Checkpoint architecture differs from its factorial arm")
    head = model.model[-1]
    points = [layer.cross_attn.n_points for layer in head.decoder.layers]
    if (head.num_queries, head.num_decoder_layers, head.decoder.eval_idx, head.f, points) != (300, 3, 2, [16, 23, 22], [4, 4, 4]):
        raise ValueError("Decoder budget changed")
    return {"variant": variant, "queries": 300, "points_per_level": 4, "decoder_layers": 3,
            "p2_injection": variant in ("c2", "c2-la"), "p5_na7_rpb": variant in ("la", "c2-la")}


def build_model(cfg, nc=10, ch=3, seed=0, weights=None, verbose=False):
    """Copy all common layers, plus identical P2 and LA slots across relevant arms.

    This also reproduces the historical C0/C2 seed initialization. The LA slot
    uses its own seed; historical R2b accuracy is context, not a paired result.
    """
    if weights is not None:
        raise ValueError("This entry point is from scratch only; no checkpoint transfer/resume")
    variant = identify(cfg)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        base = RTDETRDetectionModel(str(MODELS["c0"]), nc=nc, ch=ch, verbose=False)
        torch.manual_seed(seed + 1000)
        p2 = RTDETRDetectionModel(str(MODELS["c2"]), nc=nc, ch=ch, verbose=False)
        torch.manual_seed(seed + 2000)
        la = RTDETRDetectionModel(str(MODELS["la"]), nc=nc, ch=ch, verbose=False)
        torch.manual_seed(seed + 3000)
        model = RTDETRDetectionModel(cfg, nc=nc, ch=ch, verbose=verbose)
        model.nc = nc  # Direct loss/backward checks before trainer attributes are attached.
        reference = base.state_dict()
        shared = {k: v for k, v in reference.items() if not k.startswith(("model.6.", "model.14."))}
        p5_state = (la if variant in ("la", "c2-la") else base).model[6].state_dict()
        p2_state = (p2 if variant in ("c2", "c2-la") else base).model[14].state_dict()
        expected = {**shared, **{"model.6." + k: v for k, v in p5_state.items()},
                    **{"model.14." + k: v for k, v in p2_state.items()}}
        model.load_state_dict(expected, strict=True)
        actual = model.state_dict()
        if any(not torch.equal(actual[k], v) for k, v in expected.items()):
            raise RuntimeError("Factorial initialization verification failed")
        model.ablation_initialization = {
            "seed": seed, "p2_seed": seed + 1000, "p5_la_seed": seed + 2000,
            "source": "from scratch, independently seeded reference modules; no pretrained weights",
            "shared_state_sha256": tensor_digest(shared), "shared_state_verified": True,
            "p2_slot_sha256": tensor_digest(p2_state), "p5_slot_sha256": tensor_digest(p5_state),
            "r0_full_state_sha256": tensor_digest(reference), "variant": variant,
            "excluded_slots": ["model.6: AIFI or NAIFI", "model.14: P3 projection or P2 injection"],
        }
    inspect_model(model, variant)
    return model


class FactorialTrainer(ContinuationTrainer):
    """Use paired initialization in the actual trainer rebuild; preserve resume.pt."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        return build_model(cfg, self.data["nc"], self.data["channels"], self.args.seed, weights, verbose)


class FactorialHealth(SFIHealth):
    def start(self, trainer):
        super().start(trainer)
        p5 = unwrap_model(trainer.model).model[6]
        if isinstance(p5, NAIFI):
            for name, parameter in (("p5_qkv_grad_abs", p5.attn.qkv.weight),
                                    ("p5_rpb_grad_abs", p5.attn.relative_position_bias)):
                self.handles.append(parameter.register_hook(
                    lambda grad, label=name: self._sample(label, grad.float().abs().mean())))
