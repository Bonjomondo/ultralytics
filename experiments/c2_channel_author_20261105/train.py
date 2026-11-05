"""Paired author-engine runs with diagnostics and auditable raw-state recovery."""
import argparse
from collections import OrderedDict
from copy import deepcopy
import csv
import json
import os
import pickle
import random
import time

import numpy as np
import torch
from ultralytics import __version__
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.nn.modules.p2_channel_gate import P2GateConcat
from ultralytics.utils import callbacks
from ultralytics.utils.torch_utils import de_parallel

from common import ARMS, CONFIGS, DATA, EXP, RECIPE, RUNS, now, run_name, sha, write


def state_sha(state):
    h = __import__("hashlib").sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode() + str(v.dtype).encode() + str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def paired(cfg, seed=0):
    devices = [0] if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        base = RTDETRDetectionModel(str(CONFIGS["c2"]), nc=10, verbose=False)
        torch.manual_seed(seed + 1000)
        model = RTDETRDetectionModel(cfg, nc=10, verbose=False)
        ref = base.state_dict()
        current = model.state_dict()
        assert all(k in current and v.shape == current[k].shape for k, v in ref.items())
        model.load_state_dict(ref, strict=False)
        assert all(torch.equal(model.state_dict()[k], v) for k, v in ref.items())
        model.initialization_receipt = {"seed": seed, "shared_sha256": state_sha(ref),
            "shared_items": len(ref), "shared_verified": True, "no_trained_checkpoint": True,
            "extra_keys": sorted(set(current) - set(ref)), "identity_gate": True}
    return model


def cpu_tree(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: cpu_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [cpu_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(cpu_tree(v) for v in x)
    return x


def atomic_checkpoint(path, c):
    tmp = path.with_suffix(".pt.tmp")
    with tmp.open("wb") as f:
        torch.save(c, f, pickle_module=pickle)
        f.flush()
        os.fsync(f.fileno())
    loaded = torch.load(tmp, map_location="cpu", pickle_module=pickle)
    assert loaded["epoch"] == c["epoch"] and loaded["optimizer"] is not None
    assert all(torch.isfinite(v).all() for v in loaded["ema"].state_dict().values())
    del loaded
    tmp.replace(path)


class PairedTrainer(RTDETRTrainer):
    def check_resume(self, overrides):
        self.resume = bool(self.args.resume)
        if self.resume:
            assert str(self.args.resume).endswith("last.pt")

    def get_model(self, cfg=None, weights=None, verbose=True):
        assert weights is None, "Implicit EMA warm start is forbidden"
        model = paired(cfg, self.args.seed)
        write(self.save_dir / "initialization.json", model.initialization_receipt)
        return model

    def setup_model(self):
        if not self.resume:
            return super().setup_model()
        c = torch.load(self.args.resume, map_location="cpu", weights_only=False)
        assert c["train_args"]["seed"] == self.args.seed
        self.model = RTDETRDetectionModel(c["model"].yaml, nc=10, verbose=False)
        self.model.load_state_dict(c["model"].float().state_dict(), strict=True)
        assert all(torch.equal(v, c["model"].state_dict()[k]) for k, v in self.model.state_dict().items())
        return c

    def resume_training(self, c):
        super().resume_training(c)
        if c is None:
            return
        self.stopper.__dict__.update(c["stopper"])
        self.scheduler.load_state_dict(c["scheduler"])
        self.scaler.load_state_dict(c["scaler"])
        restored = self.optimizer.state_dict()
        for k, state in c["optimizer"]["state"].items():
            for n, v in state.items():
                got = restored["state"][k][n]
                assert torch.equal(v.cpu(), got.cpu()) if isinstance(v, torch.Tensor) else v == got
        assert all(torch.equal(v.cpu(), c["ema"].state_dict()[k].cpu()) for k, v in self.ema.ema.state_dict().items())
        assert self.ema.updates == c["updates"] and self.start_epoch == c["epoch"] + 1
        self.restored_rng = c["rng"]
        write(self.save_dir / f"restore_epoch{self.start_epoch}.json", {
            "raw_exact": True, "ema_exact": True, "optimizer_exact": True,
            "checkpoint_sha256": sha(self.args.resume), "completed_epochs": self.start_epoch,
            "limits": "epoch-boundary replay; loader order may differ, not bitwise identical continuation"})

    def attach_diagnostics(self, trainer):
        self.capture_features = False
        self.feature_samples = []
        self.grad_samples = []
        self.batches_in_epoch = 0
        raw = de_parallel(self.model)
        self.feature_handle = raw.model[17].register_forward_hook(self.observe_features)
        if hasattr(self, "restored_rng"):
            rng = self.restored_rng
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state_all(rng["cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])

    def alpha_values(self):
        gate = de_parallel(self.model).model[18]
        return gate.alpha().detach().float() if isinstance(gate, P2GateConcat) else torch.ones(1, device=self.device)

    def observe_features(self, module, inputs, output):
        if self.capture_features and module.training:
            x = output.detach().float()
            a = self.alpha_values().view(1, -1, 1, 1)
            self.feature_samples.append([float(x.square().mean().sqrt()), float((x * a).square().mean().sqrt())])

    def epoch_start(self, trainer):
        self.batches_in_epoch = 0
        self.feature_samples = []
        self.grad_samples = []
        self.epoch_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    def batch_start(self, trainer):
        self.capture_features = self.batches_in_epoch < 3

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if self.epoch == 0 and self.batches_in_epoch == 0:
            self.first_batch_receipt = {"images": [str(p) for p in batch["im_file"]],
                "image_sha256": state_sha({"img": batch["img"]}),
                "targets_sha256": state_sha({k: batch[k] for k in ["cls", "bboxes", "batch_idx"]})}
        return batch

    def optimizer_step(self):
        if self.capture_features:
            raw = de_parallel(self.model)
            p2grad = raw.model[17].conv.weight.grad
            gate = raw.model[18]
            gategrad = gate.logits.grad if isinstance(gate, P2GateConcat) else None
            self.grad_samples.append([float(p2grad.detach().float().norm()) if p2grad is not None else 0.,
                                      float(gategrad.detach().float().norm()) if gategrad is not None else 0.])
        return super().optimizer_step()

    def batch_end(self, trainer):
        if not torch.isfinite(self.loss).all():
            raise FloatingPointError("Nonfinite training loss")
        assert self.accumulate == 1 and self.batch_size == 4 and not self.amp
        if self.epoch == 0 and self.batches_in_epoch == 0:
            write(self.save_dir / "first_batch.json", {**self.first_batch_receipt,
                "loss_items": self.loss_items.detach().cpu().tolist(), "accumulate": self.accumulate})
        self.batches_in_epoch += 1
        self.capture_features = False
        if self.batches_in_epoch == 1 or self.batches_in_epoch % 100 == 0:
            self.progress(False)

    def progress(self, end):
        a = self.alpha_values()
        write(self.save_dir / "status.json", {"status": "training", "pid": os.getpid(),
            "current_epoch": self.epoch + 1, "completed_epochs": self.epoch + 1 if end else self.epoch,
            "batch_in_epoch": self.batches_in_epoch, "updated_utc": now(), "workers": 0,
            "alpha_mean": float(a.mean()), "native_metrics": self.metrics, "accumulate": self.accumulate})

    def fit_end(self, trainer):
        self.progress(True)
        a = self.alpha_values().cpu().tolist()
        row = {"epoch": self.epoch + 1, "alpha": a, "mean": float(np.mean(a)), "std": float(np.std(a)),
            "min": min(a), "max": max(a), "below_0_9": sum(v < .9 for v in a),
            "above_1_1": sum(v > 1.1 for v in a), "low_saturation": sum(v < .1 for v in a),
            "high_saturation": sum(v > 1.9 for v in a),
            "p2_feature_rms_before_after": np.mean(self.feature_samples, axis=0).tolist() if self.feature_samples else [],
            "p2_gate_grad_norm": np.mean(self.grad_samples, axis=0).tolist() if self.grad_samples else [],
            "sampling": "first three training batches; gradient norms before clipping, FP32",
            "native_metrics": self.metrics, "epoch_seconds": time.perf_counter() - self.epoch_started,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30}
        with (self.save_dir / "diagnostics.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")

    def save_model(self):
        raw = de_parallel(self.model)
        hooks = raw.model[17]._forward_hooks
        raw.model[17]._forward_hooks = OrderedDict()
        try:
            m = deepcopy(raw).cpu().float()
        finally:
            raw.model[17]._forward_hooks = hooks
        ema = deepcopy(self.ema.ema).cpu().float()
        for model in (m, ema):
            if hasattr(model, "criterion"):
                model.criterion = None
        rows = list(csv.DictReader(self.csv.open()))
        c = dict(epoch=self.epoch, best_fitness=self.best_fitness, model=m, ema=ema,
            updates=self.ema.updates, optimizer=cpu_tree(self.optimizer.state_dict()),
            scaler=self.scaler.state_dict(), scheduler=self.scheduler.state_dict(), stopper=vars(self.stopper).copy(),
            train_args=vars(self.args), train_metrics={**self.metrics, "fitness": self.fitness},
            train_results={k.strip(): [float(r[k]) for r in rows] for k in rows[0]},
            date=now(), version=__version__, rng={"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(), "numpy": np.random.get_state(), "python": random.getstate()},
            source_manifest_sha256=os.environ.get("AUTHOR_GATE_MANIFEST_SHA"))
        atomic_checkpoint(self.last, c)
        if self.best_fitness == self.fitness:
            shutil = __import__("shutil")
            tmp = self.best.with_suffix(".pt.tmp")
            shutil.copyfile(self.last, tmp)
            tmp.replace(self.best)

    def final_eval(self):
        pass  # Independent FP32 eval follows; preserve complete checkpoints.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(8)
    callbacks.add_integration_callbacks = lambda instance: None
    if args.smoke:
        import yaml
        data = yaml.safe_load(DATA.read_text())
        for split, count in [("train", 32), ("val", 16)]:
            images = sorted((__import__("pathlib").Path(data["path"]) / data[split]).glob("*.jpg"))[:count]
            assert len(images) == count
            listing = RUNS / "smoke_data" / f"{split}.txt"
            listing.parent.mkdir(parents=True, exist_ok=True)
            listing.write_text("".join(str(p) + "\n" for p in images))
            data[split] = str(listing)
        (RUNS / "smoke.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    run = RUNS / run_name(args.arm, args.seed, args.epochs, args.smoke)
    checkpoint = run / "weights/last.pt"
    if checkpoint.exists() and not args.resume:
        raise RuntimeError("Existing run; refusing to overwrite")
    recipe = {**RECIPE, "epochs": 1 if args.smoke else args.epochs, "seed": args.seed,
        "model": str(CONFIGS[args.arm]), "data": str(RUNS / "smoke.yaml" if args.smoke else DATA),
        "project": str(RUNS), "name": run.name, "exist_ok": True, "plots": not args.smoke}
    if args.resume:
        assert checkpoint.exists()
        c = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert c["train_args"]["epochs"] == recipe["epochs"]
        assert c["source_manifest_sha256"] == os.environ.get("AUTHOR_GATE_MANIFEST_SHA")
        # Rewind only nondurable curve rows, preserving the original on the server.
        import shutil
        for filename in ["results.csv", "diagnostics.jsonl"]:
            path = run / filename
            if path.exists():
                shutil.copyfile(path, path.with_name(filename + f".before_resume_{int(time.time())}"))
                lines = path.read_text().splitlines(True)
                keep = c["epoch"] + 1 + (1 if filename == "results.csv" else 0)
                path.write_text("".join(lines[:keep]))
        recipe.update(resume=str(checkpoint), model=str(checkpoint))
        del c
    t = PairedTrainer(overrides=recipe)
    for event, method in [("on_train_start", t.attach_diagnostics), ("on_train_epoch_start", t.epoch_start),
        ("on_train_batch_start", t.batch_start), ("on_train_batch_end", t.batch_end), ("on_fit_epoch_end", t.fit_end)]:
        t.add_callback(event, method)
    try:
        t.train()
        write(run / "status.json", {"status": "training_complete_pending_evaluation", "completed_epochs": t.epoch + 1,
              "updated_utc": now(), "workers": 0, "best_fitness": t.best_fitness})
    except BaseException as error:
        write(run / "failure.json", {"error": repr(error), "updated_utc": now(), "checkpoint_exists": checkpoint.exists()})
        raise


if __name__ == "__main__":
    main()
