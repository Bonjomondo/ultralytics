"""Export FP32 query predictions; CPU COCOeval runs in a separate process."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/zjnu/文档/ZHJ_WorkSpace/reproductions/rtdetr-r18-sodetr-paper-20261027')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda: f.read(1024*1024), b''):
            h.update(x)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    out = run / 'evaluation'
    out.mkdir(exist_ok=True)
    weights = run / 'weights/best.pt'
    audit = json.loads((ROOT / 'dataset_audit.json').read_text())
    gt_path = Path(audit['coco_gt']['path'])
    assert sha(gt_path) == audit['coco_gt']['sha256']
    gt = json.loads(gt_path.read_text())
    ids = {Path(im['file_name']).stem: im['id'] for im in gt['images']}
    if args.smoke:
        wanted = {Path(x).stem for x in (BASE / 'data/val_smoke.txt').read_text().splitlines()}
        selected = {v for k, v in ids.items() if k in wanted}
    else:
        selected = set(ids.values())
    assert len(selected) == (16 if args.smoke else 548)

    class ArchiveValidator(RTDETRValidator):
        def pred_to_json(self, predn, filename):
            im_id = ids[Path(filename).stem]
            for x1, y1, x2, y2, score, cls in predn.detach().cpu().tolist():
                self.jdict.append({'image_id': im_id, 'category_id': int(cls)+1,
                                   'bbox': [x1, y1, x2-x1, y2-y1], 'score': score})

    from ultralytics.utils import callbacks
    callbacks.add_integration_callbacks = lambda instance: None
    model = RTDETR(str(weights))
    ckpt_epoch = int(model.ckpt['epoch'])+1
    metric = model.val(validator=ArchiveValidator, data=str(ROOT / ('data-smoke.yaml' if args.smoke else 'data.yaml')),
                       device='0', batch=4, imgsz=640, half=False, workers=0, save_json=True,
                       plots=not args.smoke, project=str(run), name='evaluation', exist_ok=True,
                       conf=.001, max_det=300)
    pred = out / 'predictions.json'
    boxes = json.loads(pred.read_text())
    assert {x['image_id'] for x in boxes} == selected
    assert len(boxes) == 300 * len(selected)
    assert all(math.isfinite(v) for x in boxes for v in x['bbox']+[x['score']])
    meta = {'checkpoint': str(weights), 'checkpoint_sha256': sha(weights), 'checkpoint_epoch': ckpt_epoch,
            'gt_path': str(gt_path), 'gt_sha256': sha(gt_path), 'prediction_sha256': sha(pred),
            'images': len(selected), 'detections': len(boxes), 'native_metrics': metric.results_dict,
            'protocol': 'author RTDETRValidator, FP32, all 300 argmax queries/image, no NMS or confidence threshold, unrounded; same as B0'}
    (out / 'prediction_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float)+'\n')
    print(json.dumps(meta, ensure_ascii=False, indent=2, default=float))


if __name__ == '__main__':
    torch.set_num_threads(8)
    main()
