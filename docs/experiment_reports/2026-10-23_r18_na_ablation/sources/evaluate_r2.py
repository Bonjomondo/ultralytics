"""Native FP32 validation and the same formal COCO AP protocol as the completed R0/R1 pair."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import RTDETR


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=['r2a', 'r2b', 'r2c'])
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = Path(__file__).resolve().parent
    weight = args.run / 'weights/best.pt'
    annotation = Path('/home/zjnu/文档/ZHJ_WorkSpace/datasets/VisDrone2019-DET-YOLO/annotations/VisDrone2019-DET_val_coco.json')
    baseline = json.loads((output.parent / 'completed_pair/r0_formal_metrics.json').read_text())
    assert digest(annotation) == baseline['protocol']['annotation_sha256']
    with (args.run / 'results.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 150 and int(rows[-1]['epoch']) == 150
    torch.set_num_threads(4)
    model = RTDETR(str(weight))
    result = model.val(data=str(root / 'configs/visdrone-local.yaml'), device='0', imgsz=640,
                       batch=2, workers=0, half=False, plots=False, verbose=False, save_json=True,
                       project=str(args.run), name='formal_native', exist_ok=False)
    prediction_path = Path(result.save_dir) / 'predictions.json'
    predictions = json.loads(prediction_path.read_text())
    gt = COCO(str(annotation))
    images = {Path(im['file_name']).name: im['id'] for im in gt.dataset['images']}
    for row in predictions:
        row['image_id'] = images[row['file_name']]
    evaluator = COCOeval(gt, gt.loadRes(predictions), 'bbox')
    evaluator.params.imgIds = sorted(gt.getImgIds())
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl')
    metrics = dict(zip(names, map(float, evaluator.stats)))
    protocol = {'iou_type': 'bbox', 'image_ids': [int(i) for i in evaluator.params.imgIds], 'max_dets': [1, 10, 100],
                'engine': 'pycocotools.COCOeval', 'engine_version': version('pycocotools'),
                'iou_thresholds': evaluator.params.iouThrs.tolist(), 'area': 'all',
                'annotation_sha256': digest(annotation), 'category_ids': [int(i) for i in sorted(gt.cats)]}
    assert protocol == baseline['protocol'], 'Evaluation protocol differs from R0'
    record = {'variant': args.variant, 'finished_utc': datetime.now(timezone.utc).isoformat(),
              'checkpoint': str(weight), 'checkpoint_sha256': digest(weight),
              'native_internal_metrics': {key: float(value) for key, value in result.results_dict.items()}, 'predictions_path': str(prediction_path),
              'predictions_file_sha256': digest(prediction_path), 'metrics': metrics, 'protocol': protocol,
              'delta_vs_r0_pp': {name: (metrics[name] - baseline['metrics'][name]) * 100 for name in names}}
    (output / f'{args.variant}_formal_metrics.json').write_text(json.dumps(record, indent=2) + '\n')
    comparisons = []
    for variant in ('r0', 'r1', 'r2a', 'r2b', 'r2c'):
        path = (output.parent / f'completed_pair/{variant}_formal_metrics.json'
                if variant in ('r0', 'r1') else output / f'{variant}_formal_metrics.json')
        if path.exists():
            value = json.loads(path.read_text())
            comparisons.append({'variant': variant, **value['metrics']})
    with (output / 'formal_metrics_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['variant', *names])
        writer.writeheader()
        writer.writerows(comparisons)
    print('FORMAL EVALUATION COMPLETE', args.variant, metrics, flush=True)


if __name__ == '__main__':
    main()
