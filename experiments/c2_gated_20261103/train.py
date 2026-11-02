"""Train paired C2 fusion screens; all non-gate initialization is exactly shared."""
import argparse
import csv
import math
import os
import random
from copy import deepcopy

import numpy as np
import torch

from common import ARMS, CONFIGS, ROOT, RUNS, atomic_save, now, write
from experiments.r18_p2_sfi.train import SFIHealth, build_model, tensor_digest
from train_rtdetr_r18_na import RECIPE
from ultralytics import RTDETR, __version__
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules.gated_fusion import C2GatedConcat
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import unwrap_model


def paired(cfg, nc=10, ch=3, seed=0, verbose=False):
    with torch.random.fork_rng(devices=[]):
        reference = build_model(str(CONFIGS["c2"]), nc=nc, ch=ch, seed=seed)
        torch.manual_seed(seed + 2000)
        model = RTDETRDetectionModel(cfg, nc=nc, ch=ch, verbose=verbose)
        ref, dst = reference.state_dict(), model.state_dict()
        assert all(k in dst and dst[k].shape == v.shape for k, v in ref.items())
        model.load_state_dict(ref, strict=False)
        assert all(torch.equal(model.state_dict()[k], v) for k, v in ref.items())
        model.ablation_initialization = {
            "method": "fresh seeded C2; exact copy of ALL C2 tensors including P2 branch; no pretrained checkpoint",
            "seed": seed, "shared_state_sha256": tensor_digest(ref),
            "shared_state_items": len(ref), "shared_state_verified": True,
            "new_gate_state": sorted(set(dst) - set(ref)),
            "identity_gate_initialization": True,
        }
    return model


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(v) for v in value)
    return value


class GatedTrainer(RTDETRTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is not None:
            raise RuntimeError("Use a separately audited raw-state resume; implicit EMA warm starts are disabled")
        return paired(cfg, self.data["nc"], self.data["channels"], self.args.seed, verbose)

    def save_model(self):
        raw = deepcopy(unwrap_model(self.model)).cpu().float()
        ema = deepcopy(self.ema.ema).cpu().float()
        for model in (raw, ema):
            for module in model.modules():
                module._forward_hooks.clear()
                module._forward_pre_hooks.clear()
                module._backward_hooks.clear()
                if isinstance(module, C2GatedConcat):
                    module.stats = {}
                    module.capture = False
            for param in model.parameters():
                if param._backward_hooks:
                    param._backward_hooks.clear()
            if hasattr(model, "criterion"):
                model.criterion = None
        ckpt = {
            "epoch": self.epoch, "best_fitness": self.best_fitness,
            "model": raw, "ema": ema, "updates": self.ema.updates,
            "optimizer": cpu_state(self.optimizer.state_dict()), "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict(), "train_args": vars(self.args),
            "train_metrics": {**self.metrics, "fitness": self.fitness},
            "train_results": self.read_results_csv(), "date": now(), "version": __version__,
            "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                    "numpy": np.random.get_state(), "python": random.getstate()},
            "stopper": vars(self.stopper).copy(),
            "save_protocol": "FP32 CPU raw+EMA+optimizer+RNG, stdlib pickle, verified atomic replacement",
        }
        atomic_save(self.last, ckpt)
        if self.best_fitness == self.fitness:
            atomic_save(self.best, ckpt)

    def final_eval(self):
        # Keep complete last/best checkpoints. Independent evaluation is a queue stage.
        pass


class GateHealth(SFIHealth):
    def start(self, trainer):
        super().start(trainer)
        self.gate = next((m for m in trainer.model.modules() if isinstance(m, C2GatedConcat)), None)
        if self.gate is not None:
            self.handles.append(self.gate.register_forward_hook(self.observe))
            for name, param in self.gate.named_parameters():
                self.handles.append(param.register_hook(
                    lambda grad, n=name: self._sample("gate_grad_" + n, grad.float().abs().mean())))

    def observe(self, module, inputs, output):
        if self.capture:
            for name, value in module.stats.items():
                self._sample(name, value)

    def batch_start(self, trainer):
        super().batch_start(trainer)
        if self.gate is not None:
            self.gate.capture = self.capture


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    from ultralytics.utils import callbacks
    callbacks.add_integration_callbacks = lambda instance: None
    run = RUNS / ("smoke_" + args.arm if args.smoke else args.arm)
    if (run / "weights/last.pt").exists():
        raise RuntimeError(f"Refusing to overwrite existing run: {run}")
    recipe = {**RECIPE, "epochs": 1 if args.smoke else 150,
              "patience": 0, "batch": 8, "amp": False, "deterministic": False,
              "seed": 0, "plots": not args.smoke}
    data = ROOT / ("experiments/c2_gated_20261103/data-smoke.yaml" if args.smoke else "configs/visdrone-local.yaml")
    model = RTDETR(str(CONFIGS[args.arm]))
    GateHealth().attach(model)
    state = {"status": "starting", "arm": args.arm, "pid": os.getpid(),
             "started_utc": now(), "completed_epochs": 0, "recipe": recipe}
    write(run / "status.json", state)
    batches = 0

    def progress(trainer, end=False):
        nonlocal batches
        if not end:
            batches += 1
            if batches % 50 and batches != 1:
                return
        state.update(status="training", current_epoch=trainer.epoch + 1,
                     completed_epochs=trainer.epoch + 1 if end else state["completed_epochs"],
                     observed_batches=batches, updated_utc=now(),
                     metrics={k: float(v) for k, v in trainer.metrics.items()})
        write(run / "status.json", state)

    model.add_callback("on_train_batch_end", progress)
    model.add_callback("on_fit_epoch_end", lambda t: progress(t, True))
    try:
        model.train(trainer=GatedTrainer, **recipe, data=str(data), device="0", workers=0,
                    project=str(RUNS), name=run.name, pretrained=False, exist_ok=True)
        state.update(status="training_complete_pending_evaluation", updated_utc=now())
        write(run / "status.json", state)
    except BaseException as error:
        state.update(status="failed", error=repr(error), updated_utc=now())
        write(run / "status.json", state)
        raise


if __name__ == "__main__":
    main()
