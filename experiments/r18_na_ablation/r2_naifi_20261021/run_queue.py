"""Run three full R2 ablations serially, evaluating each completed checkpoint."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[2]
RUNS = ROOT / 'runs/detect/r18_na_ablation'
STATUS = OUTPUT / 'queue_status.json'
VARIANTS = ['r2a', 'r2b', 'r2c']


def now():
    return datetime.now(timezone.utc).isoformat()


def save(record):
    record['updated_utc'] = now()
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(STATUS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='Resume recorded completed phases without retraining them')
    args = parser.parse_args()
    queue_lock = (OUTPUT / 'queue.lock').open('a')
    fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.chdir(ROOT)
    os.environ['PYTHONPATH'] = str(ROOT)
    os.environ['MPLBACKEND'] = 'Agg'
    state = {'status': 'running', 'started_utc': now(), 'pid': os.getpid(),
             'order': VARIANTS, 'completed_training': [], 'completed_evaluation': []}
    sources = [ROOT / name for name in ('train_rtdetr_r18_na.py', 'ultralytics/nn/modules/rtdetr_na.py',
               'ultralytics/nn/modules/__init__.py', 'ultralytics/nn/tasks.py')]
    sources += sorted((ROOT / 'ultralytics/cfg/models/rt-detr').glob('rtdetr-resnet18*.yaml'))
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest = OUTPUT / 'source_sha256.json'
    if args.resume:
        state = json.loads(STATUS.read_text())
        if state['order'] != VARIANTS:
            raise RuntimeError('Recorded variant order differs from this queue')
        if json.loads(manifest.read_text()) != source_hashes:
            raise RuntimeError('Training sources differ from the original queue manifest')
        state.setdefault('resume_history', []).append({'resumed_utc': now(), 'previous_pid': state.get('pid'),
                                                      'previous_error': state.get('error')})
        state.update(status='running', pid=os.getpid(), child_pid=None)
        state.pop('error', None)
        state.pop('finished_utc', None)
    else:
        if STATUS.exists():
            raise FileExistsError('Queue state already exists; use --resume to continue it')
        manifest.write_text(json.dumps(source_hashes, indent=2) + '\n')
    save(state)
    try:
        for variant in VARIANTS:
            name = f'rtdetr-r18-{variant}-150e-s0-20261021'
            run = RUNS / name
            if variant in state['completed_training']:
                with (run / 'results.csv').open() as file:
                    rows = list(csv.DictReader(file))
                if len(rows) != 150 or int(rows[-1]['epoch']) != 150:
                    raise RuntimeError(f'Recorded training is incomplete: {run}')
                for checkpoint in ('best.pt', 'last.pt'):
                    if not (run / 'weights' / checkpoint).is_file():
                        raise FileNotFoundError(run / 'weights' / checkpoint)
            elif run.exists():
                raise FileExistsError(f'Refusing to overwrite existing run: {run}')
            for path, expected in source_hashes.items():
                if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f'Source changed during queue: {path}')
            commands = [
                ('training', [sys.executable, '-u', 'train_rtdetr_r18_na.py', '--variant', variant,
                 '--epochs', '150', '--batch', '8', '--imgsz', '640', '--seed', '0', '--amp', 'False',
                 '--workers', '0', '--device', '0', '--project', str(RUNS), '--name', name]),
                ('evaluation', [sys.executable, '-u', str(OUTPUT / 'evaluate_r2.py'),
                 '--variant', variant, '--run', str(run)]),
            ]
            for phase, command in commands:
                if variant in state[f'completed_{phase}']:
                    if phase == 'evaluation':
                        record = json.loads((OUTPUT / f'{variant}_formal_metrics.json').read_text())
                        checkpoint_hash = hashlib.sha256((run / 'weights/best.pt').read_bytes()).hexdigest()
                        if record['checkpoint_sha256'] != checkpoint_hash:
                            raise RuntimeError(f'Evaluated checkpoint changed: {run}')
                    print(now(), 'skipping completed', phase, variant, flush=True)
                    continue
                for path, expected in source_hashes.items():
                    if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                        raise RuntimeError(f'Source changed during queue: {path}')
                log = ROOT / 'logs' / f'{name}-{phase}.log'
                if args.resume and log.exists():
                    log = log.with_name(f'{name}-{phase}-resume-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.log')
                state.update(variant=variant, phase=phase, run=str(run), log=str(log), command=command)
                save(state)
                print(now(), phase, variant, 'log=', log, flush=True)
                with log.open('x') as file:
                    child = subprocess.Popen(command, stdout=file, stderr=subprocess.STDOUT)
                    state['child_pid'] = child.pid
                    save(state)
                    rc = child.wait()
                if rc:
                    raise RuntimeError(f'{variant} {phase} exited {rc}; inspect {log}')
                state[f'completed_{phase}'].append(variant)
                save(state)
        state.update(status='completed', phase='done', finished_utc=now(), child_pid=None)
        save(state)
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        save(state)
        raise


if __name__ == '__main__':
    main()
