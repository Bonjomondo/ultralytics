"""Frozen experiment identity and shared author-recipe settings."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parent
REPO = Path(os.environ.get('AUTHOR_P2_REPO', EXP.parents[1]))
BACKEND = EXP / 'backend'
if not BACKEND.exists():
    BACKEND = REPO / 'experiments/c2_channel_author_20261105/backend'
RUNS = REPO / 'runs/p2_full_author_20261107'
REPORT = REPO / 'docs/experiment_reports/2026-11-07_p2_full_author'
BRANCH = 'feat/r18-p2-full-author'
ARMS = ('b0', 'c2', 'full')
SEED = 1
MILESTONES = (150, 250, 350)
CONFIGS = {arm: EXP / 'configs' / f'{arm}.yaml' for arm in ARMS}
DATA = EXP / 'configs/data.yaml'
GT = REPO.parent / 'datasets/VisDrone2019-DET-YOLO/annotations/VisDrone2019-DET_val_coco.json'
GT_SHA = '6f57aa4b42d64ee7b99b8a6a5b159ad19cba2eacf00bd86f23314966614bf4d8'
RECIPE = dict(epochs=350, patience=0, batch=4, imgsz=640, optimizer='AdamW',
              lr0=.0001, lrf=1., momentum=.9, weight_decay=.0001,
              warmup_epochs=2000, warmup_bias_lr=.1, warmup_momentum=.8,
              nbs=4, mixup=.2, mosaic=1., close_mosaic=0, amp=False,
              deterministic=True, seed=SEED, pretrained=False, max_det=300,
              cache=False, device='0', workers=0, val=True, plots=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run_name(arm, seed=SEED, epochs=350, smoke=False):
    return f"{'smoke' if smoke else 'main'}_{arm}_{epochs}e_s{seed}"


def dataset_identity():
    import yaml
    data = yaml.safe_load(DATA.read_text())
    result = {'gt_sha256': sha(GT), 'data_yaml_sha256': sha(DATA)}
    assert result['gt_sha256'] == GT_SHA
    for split, count in [('train', 6471), ('val', 548)]:
        images = sorted((Path(data['path']) / data[split]).glob('*.jpg'))
        assert len(images) == count
        labels, names = hashlib.sha256(), hashlib.sha256()
        for path in images:
            names.update(f'{path.name}:{path.stat().st_size}\n'.encode())
            label = path.parent.parent / 'labels' / (path.stem + '.txt')
            labels.update(path.name.encode() + b'\0' + label.read_bytes())
        result[split] = {'images': count, 'labels_sha256': labels.hexdigest(),
                         'image_names_sizes_sha256': names.hexdigest()}
    return result
