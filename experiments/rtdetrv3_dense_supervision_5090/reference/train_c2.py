"""Train migrated C2 with the original B0 recipe and paired random initialization."""
import argparse
import csv
import hashlib
import json
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/zjnu/文档/ZHJ_WorkSpace/reproductions/rtdetr-r18-sodetr-paper-20261027')
NAME = 'rtdetr-r18-c2-matched-b0-350e-s0'
RECIPE = dict(epochs=350, patience=40, batch=4, imgsz=640, optimizer='AdamW',
              lr0=.0001, lrf=1.0, momentum=.9, weight_decay=.0001,
              warmup_epochs=2000, warmup_bias_lr=.1, warmup_momentum=.8,
              nbs=4, mixup=.2, mosaic=1.0, close_mosaic=0, amp=False,
              deterministic=True, seed=0, pretrained=False, max_det=300,
              cache=False, device='0', workers=8, val=True, plots=True,
              project=str(ROOT / 'runs'), name=NAME)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str)+'\n')
    tmp.replace(path)


def state_hash(state, keys):
    h = hashlib.sha256()
    for key in sorted(keys):
        t = state[key].detach().cpu().contiguous()
        h.update(key.encode()+b'\0'+str(t.dtype).encode()+b'\0'+str(tuple(t.shape)).encode()+b'\0')
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class PairedTrainer(RTDETRTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        if weights is not None or self.args.resume:
            return model
        # The B0 training constructor reset torch's CPU seed to 0 immediately
        # before get_model. Recreate that random, untrained initialization.
        # B0 did not save its initial state, so the replay cannot be verified
        # against an archived initial hash. Never load B0 trained checkpoints.
        with torch.random.fork_rng(devices=[0] if torch.cuda.is_available() else []):
            torch.manual_seed(int(self.args.seed) + 1 + RANK)
            baseline = RTDETRDetectionModel(str(BASE / 'rtdetr-resnet18-baseline.yaml'),
                                             nc=self.data['nc'], verbose=False)
        ref, target = baseline.state_dict(), model.state_dict()
        common = [k for k, v in target.items() if not k.startswith('model.17.')
                  and k in ref and v.shape == ref[k].shape]
        natural_matching = [k for k in common if torch.equal(target[k], ref[k])]
        with torch.no_grad():
            for key in common:
                target[key].copy_(ref[key])
        assert all(torch.equal(target[k], ref[k]) for k in common)
        info = {
            'seed': int(self.args.seed),
            'method': 'replayed B0 seed-zero untrained constructor and copied same-name same-shape common tensors; changed P2 branch retains independently initialized candidate tensors',
            'baseline_initial_state_was_archived': False,
            'trained_checkpoint_loaded': False,
            'baseline_config_sha256': sha(BASE / 'rtdetr-resnet18-baseline.yaml'),
            'candidate_config_sha256': sha(ROOT / 'rtdetr-resnet18-c2.yaml'),
            'common_tensors': len(common),
            'naturally_equal_before_copy': len(natural_matching),
            'common_state_sha256': state_hash(target, common),
            'baseline_replay_common_state_sha256': state_hash(ref, common),
            'changed_layer_keys': sorted(k for k in target if k.startswith('model.17.')),
            'candidate_parameters': sum(p.numel() for p in model.parameters()),
            'baseline_parameters': sum(p.numel() for p in baseline.parameters()),
        }
        assert info['common_state_sha256'] == info['baseline_replay_common_state_sha256']
        assert info['candidate_parameters'] == 20135288
        atomic_json(self.save_dir / 'initialization.json', info)
        print('INITIALIZATION='+json.dumps(info, ensure_ascii=False), flush=True)
        return model

    def final_eval(self):
        from ultralytics.utils import LOGGER
        LOGGER.info('Final FP32 predictions and separate CPU COCOeval follow in run.sh.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    recipe = dict(RECIPE)
    if args.smoke:
        recipe.update(epochs=1, patience=0, name='smoke-c2', workers=2, plots=False)
    recipe.update(data=str(ROOT / ('data-smoke.yaml' if args.smoke else 'data.yaml')), exist_ok=True)
    run = Path(recipe['project']) / recipe['name']
    run.mkdir(parents=True, exist_ok=True)
    if not args.resume and (run / 'weights/last.pt').exists():
        raise RuntimeError(f'Existing run: {run}; use --resume with an explicit checkpoint')
    if args.resume:
        resume = args.resume.resolve(strict=True)
        if resume.parent != (run / 'weights').resolve():
            raise ValueError(f'Resume checkpoint must belong to this run: {resume}')
        recipe['resume'] = str(resume)
    state = {'status': 'starting', 'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
             'recipe': recipe, 'run': str(run), 'completed_epochs': 0,
             'initialization': 'random B0 seed replay + independent C2 branch; no trained checkpoint' if not args.resume else str(args.resume)}
    started = time.time()
    atomic_json(run / 'status.json', state)
    names = ['rtdetr-resnet18-c2.yaml', 'rtdetr-resnet18-baseline.yaml', 'data.yaml',
             'data-smoke.yaml', 'dataset_audit.json', 'baseline_reference.json', 'baseline_results.csv',
             'paired_initialization_verification.json', 'dataset_current_verification.json', 'train_c2.py', 'export_predictions.py',
             'coco_eval.py', 'benchmark.py', 'finish.py', 'run.sh', 'architecture_comparison.json']
    files = sorted((ROOT / 'ultralytics').rglob('*.py')) + [ROOT / x for x in names]
    hashes = {str(f.relative_to(ROOT)): sha(f) for f in files}
    manifest_path = run / 'source_manifest.json'
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        changed = [k for k in set(previous) | set(hashes) if previous.get(k) != hashes.get(k)]
        if changed:
            raise RuntimeError(f'Frozen source differs from the original run: {changed}')
    else:
        atomic_json(manifest_path, hashes)
    torch.set_num_threads(8)
    model = RTDETR(str(ROOT / 'rtdetr-resnet18-c2.yaml'))
    from ultralytics.utils import callbacks
    callbacks.add_integration_callbacks = lambda instance: None
    observed_batches = 0

    def record(trainer, finished_epoch=False):
        nonlocal observed_batches
        if not finished_epoch:
            observed_batches += 1
            if observed_batches != 1 and observed_batches % 100:
                return
        state.update(status='training', current_epoch=trainer.epoch+1,
                     completed_epochs=(trainer.epoch+1 if finished_epoch else state['completed_epochs']),
                     observed_train_batches=observed_batches, elapsed_seconds=time.time()-started,
                     updated_utc=datetime.now(timezone.utc).isoformat(),
                     gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                     internal_metrics=getattr(trainer, 'metrics', {}),
                     loss_items=trainer.tloss.detach().cpu().tolist() if trainer.tloss is not None else None,
                     accumulate=getattr(trainer, 'accumulate', None))
        if finished_epoch:
            state['last_epoch_seconds'] = trainer.epoch_time
            print('EPOCH_STATUS='+json.dumps(state, default=str), flush=True)
        atomic_json(run / 'status.json', state)

    def snapshot(trainer):
        epoch = trainer.epoch+1
        if epoch != (1 if args.smoke else 150):
            return
        saved = {}
        for label, source in [('best', trainer.best), ('last', trainer.last)]:
            dest = run / 'weights' / f'epoch{epoch:03d}_{label}.pt'
            tmp = dest.with_suffix('.pt.tmp')
            shutil.copyfile(source, tmp)
            tmp.replace(dest)
            saved[label] = {'path': str(dest), 'bytes': dest.stat().st_size, 'sha256': sha(dest)}
        # Read the actual saved best epoch rather than infer it from rounded CSV.
        ckpt = torch.load(run / 'weights' / f'epoch{epoch:03d}_best.pt', map_location='cpu')
        saved['best']['epoch'] = int(ckpt['epoch'])+1
        saved['best']['train_metrics'] = ckpt.get('train_metrics')
        milestone = {'completed_epochs': epoch, 'snapshots': saved,
                     'current_native_metrics': trainer.metrics, 'best_fitness': float(trainer.best_fitness)}
        if not args.smoke:
            baseline_csv = ROOT / 'baseline_results.csv'
            rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(baseline_csv.open())]
            row = rows[epoch-1]
            best = max(rows[:epoch], key=lambda x: .1*x['metrics/mAP50(B)']+.9*x['metrics/mAP50-95(B)'])
            milestone['B0_native_comparison'] = {
                'disclaimer': 'internal framework metrics, not formal COCOeval; B0 epoch-150 weights were not archived',
                'B0_epoch_150_mAP50_95': row['metrics/mAP50-95(B)'],
                'B0_epoch_150_mAP50': row['metrics/mAP50(B)'],
                'B0_best_within_150_epoch': int(best['epoch']),
                'B0_best_within_150_mAP50_95': best['metrics/mAP50-95(B)'],
                'B1_current_mAP50_95': trainer.metrics['metrics/mAP50-95(B)'],
                'B1_best_within_150_mAP50_95': saved['best']['train_metrics']['metrics/mAP50-95(B)']}
        atomic_json(run / f'milestone_{epoch:03d}.json', milestone)
        print('MILESTONE='+json.dumps({'epoch': epoch, 'snapshots': saved}, ensure_ascii=False), flush=True)

    model.add_callback('on_train_batch_end', record)
    model.add_callback('on_fit_epoch_end', lambda trainer: record(trainer, True))
    model.add_callback('on_model_save', snapshot)
    try:
        model.train(trainer=PairedTrainer, **recipe)
        state.update(status='training_complete_pending_evaluation', updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(run / 'status.json', state)
    except BaseException:
        state.update(status='failed', error=traceback.format_exc(), updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(run / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
