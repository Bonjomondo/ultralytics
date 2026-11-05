"""Compute standard pycocotools COCOeval in a process that never imports torch."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/zjnu/文档/ZHJ_WorkSpace/reproductions/rtdetr-r18-sodetr-paper-20261027')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    out = args.run.resolve() / 'evaluation'
    meta = json.loads((out / 'prediction_metadata.json').read_text())
    gt_path = Path(meta['gt_path'])
    pred_path = out / 'predictions.json'
    assert sha(gt_path) == meta['gt_sha256'] and sha(pred_path) == meta['prediction_sha256']
    gt = COCO(str(gt_path))
    preds = json.loads(pred_path.read_text())
    img_ids = sorted({x['image_id'] for x in preds})
    assert len(img_ids) == (16 if args.smoke else 548)
    assert len(preds) == 300*len(img_ids)
    result = {'evaluator': 'pycocotools.COCOeval compiled bbox IoU, independent CPU process',
              'gt_sha256': meta['gt_sha256'], 'prediction_sha256': meta['prediction_sha256'],
              'checkpoint_sha256': meta['checkpoint_sha256'], 'checkpoint_epoch': meta['checkpoint_epoch'],
              'images': len(img_ids), 'detections': len(preds), 'unit': 'percent',
              'native_metrics': meta['native_metrics']}
    for maximum in (100, 300):
        ev = COCOeval(gt, gt.loadRes(str(pred_path)), 'bbox')
        ev.params.imgIds = img_ids
        ev.params.maxDets = [1, 10, maximum]
        ev.evaluate()
        ev.accumulate()
        precision = ev.eval['precision']

        def mean(v):
            v = v[v > -1]
            return float(v.mean()*100) if v.size else None

        def ap(area=0, iou=None, cat=None):
            v = precision[:, :, :, area, -1]
            if cat is not None:
                v = v[:, :, cat]
            if iou is not None:
                v = v[np.isclose(ev.params.iouThrs, iou)]
            return mean(v)

        result[f'coco_maxdet{maximum}_percent'] = {
            'AP': ap(), 'AP50': ap(iou=.5), 'AP75': ap(iou=.75),
            'APs': ap(1), 'APm': ap(2), 'APl': ap(3)}
        if maximum == 100:
            result['per_class_maxdet100'] = [
                {'category_id': int(c), 'name': gt.cats[c]['name'], 'AP': ap(cat=i),
                 'AP50': ap(iou=.5, cat=i), 'APs': ap(1, cat=i), 'APm': ap(2, cat=i), 'APl': ap(3, cat=i)}
                for i, c in enumerate(ev.params.catIds)]
        print('METRICS', maximum, json.dumps(result[f'coco_maxdet{maximum}_percent']), flush=True)
    (out / 'coco_metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
