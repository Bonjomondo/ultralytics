"""Atomic CPU FP32 checkpoints for the author engine; training math is unchanged."""
import os
import pickle
import random
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import torch
from ultralytics import __version__
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils.torch_utils import de_parallel


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


class SafeTrainer(RTDETRTrainer):
    def optimizer_step(self):
        p = de_parallel(self.model).model[18].logit
        self.gate_grad_abs = float(p.grad.detach().abs()) if p.grad is not None else None
        return super().optimizer_step()

    def save_model(self):
        ckpt = dict(epoch=self.epoch, best_fitness=self.best_fitness,
                    model=deepcopy(de_parallel(self.model)).cpu().float(),
                    ema=deepcopy(self.ema.ema).cpu().float(), updates=self.ema.updates,
                    optimizer=cpu_tree(self.optimizer.state_dict()), train_args=vars(self.args),
                    train_metrics={**self.metrics, 'fitness': self.fitness},
                    date=datetime.now(timezone.utc).isoformat(), version=__version__,
                    scheduler=self.scheduler.state_dict(), scaler=self.scaler.state_dict(),
                    stopper=vars(self.stopper).copy(),
                    rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                             numpy=np.random.get_state(), python=random.getstate()))
        import csv
        rows = list(csv.DictReader(self.csv.open()))
        ckpt['train_results'] = {k.strip(): [float(row[k]) for row in rows] for k in rows[0]} if rows else {}
        for model in (ckpt['model'], ckpt['ema']):
            if hasattr(model, 'criterion'):
                model.criterion = None
        self.wdir.mkdir(parents=True, exist_ok=True)
        for path in [self.last] + ([self.best] if self.best_fitness == self.fitness else []):
            temp = path.with_suffix('.pt.tmp')
            with temp.open('wb') as f:
                torch.save(ckpt, f, pickle_module=pickle)
                f.flush()
                os.fsync(f.fileno())
            loaded = torch.load(temp, map_location='cpu', pickle_module=pickle)
            assert loaded['epoch'] == self.epoch and loaded['optimizer'] is not None
            assert all(torch.isfinite(t).all() for t in loaded['ema'].state_dict().values())
            del loaded
            temp.replace(path)
