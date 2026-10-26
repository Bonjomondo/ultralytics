"""Collect compact final evidence, replay category COCO AP from saved predictions.

No inference or training: predictions and large COCO match lists stay on server.
The live report JSON is the sole archive of fresh state/curves/original timings.
"""
import argparse
from contextlib import redirect_stdout
import copy
import csv
import hashlib
import io
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(args):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    raw = args.output / 'raw'
    raw.mkdir(parents=True, exist_ok=False)
    old = read(args.baseline)
    baseline = old['conditions']['d3']
    entries = [('c2-q300-p4-d3', Path(baseline['predictions']['path']),
                Path(baseline['coco']['path']), baseline['metrics'], baseline['predictions']['sha256'])]
    live = read(args.live)
    records = [v for v in live['records'] if v['state']['role'] == 'fresh_150e']
    assert len(records) == 4 and live['queue']['status'] == 'completed'
    server = []
    health_rows = []
    for record in records:
        state, evaluation = record['state'], record['evaluation']
        run = Path(state['run'])
        entries.append((state['variant'], run.parent / (run.name + '-eval') / 'predictions.json',
                        run.parent / (run.name + '-eval') / 'coco.json', evaluation['metrics'],
                        evaluation['predictions_sha256']))
        for path in [run / 'results.csv', run / 'health.jsonl', run / 'experiment.json',
                     run.parent / 'logs' / (run.name + '.log'),
                     *[Path(v['path']) for v in state['checkpoints'].values()]]:
            server.append(artifact(path))
        for line in (run / 'health.jsonl').read_text().splitlines():
            health_rows.append({'variant': state['variant'], **json.loads(line)})
    annotation = Path(old['protocol']['annotation']['path'])
    assert artifact(annotation)['sha256'] == old['formal_protocol']['annotation_sha256']
    with redirect_stdout(io.StringIO()):
        gt = COCO(str(annotation))
    base_coco = read(entries[0][2])
    transitions = []
    category_rows = []
    for variant, predictions, coco_path, metrics, expected_sha in entries:
        pred_artifact = artifact(predictions)
        assert pred_artifact['sha256'] == expected_sha
        server.extend([pred_artifact, artifact(coco_path)])
        with redirect_stdout(io.StringIO()):
            dt = gt.loadRes(str(predictions))
            evaluator = COCOeval(gt, dt, 'bbox')
            evaluator.params.imgIds = sorted(gt.imgs)
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        # Replayed scalar AP must match the previously accepted full-val result.
        names = ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl')
        assert all(abs(float(value) - metrics[name]) < 1e-12 for name, value in zip(names, evaluator.stats))
        precision = evaluator.eval['precision']
        recall = evaluator.eval['recall']
        for k, category in enumerate(evaluator.params.catIds):
            row = {'variant': variant, 'category_id': category, 'category': gt.cats[category]['name'],
                   'gt_count': len(gt.getAnnIds(catIds=[category]))}
            for area, label in enumerate(('AP', 'APs', 'APm', 'APl')):
                values = precision[:, :, k, area, 2]
                valid = values[values > -1]
                row[label] = float(np.mean(valid)) if valid.size else None
            values = recall[:, k, 0, 2]
            row['AR100'] = float(np.mean(values[values > -1]))
            category_rows.append(row)
        current = read(coco_path)
        assert current['protocol'] == base_coco['protocol']
        assert current['gt_states'].keys() == base_coco['gt_states'].keys()
        if variant != entries[0][0]:
            for i, threshold in enumerate(current['protocol']['iou_thresholds']):
                groups = {}
                for gt_id, before in base_coco['gt_states'].items():
                    after = current['gt_states'][gt_id]
                    assert (before['image_id'], before['category_id']) == (after['image_id'], after['category_id'])
                    ann = gt.anns[int(gt_id)]
                    scale = 'small' if ann['area'] < 32**2 else 'medium' if ann['area'] < 96**2 else 'large'
                    a, b = before['matched'][i], after['matched'][i]
                    state = 'stable_tp' if a and b else 'lost' if a else 'gained' if b else 'stable_fn'
                    for key in [('all', 'all'), ('category', gt.cats[before['category_id']]['name']), ('size', scale)]:
                        counter = groups.setdefault(key, dict.fromkeys(('stable_tp', 'lost', 'gained', 'stable_fn'), 0))
                        counter[state] += 1
                for (group, label), counts in sorted(groups.items()):
                    transitions.append({'variant': variant, 'iou': threshold, 'group': group, 'label': label,
                                        **counts, 'net_tp': counts['gained'] - counts['lost'],
                                        'gt_count': sum(counts.values())})
        print('replayed COCO/category AP and aggregated GT:', variant, flush=True)
    csv_write(raw / 'category_ap.csv', category_rows)
    csv_write(raw / 'gt_transitions.csv', transitions)
    # Different physical depths yield different health keys; use their union.
    keys = list(dict.fromkeys(k for r in health_rows for k in r))
    csv_write(raw / 'training_health.csv', [{k: r.get(k) for k in keys} for r in health_rows])
    paired = read(args.paired)
    compact = copy.deepcopy(paired)
    compact['server_assessment'] = artifact(args.paired)
    compact['archive_policy'] = 'All CUDA samples retained; wall samples server-only; wall block/pooled summaries retained.'
    for condition in compact['conditions'].values():
        for block in condition['blocks']:
            block.pop('wall_ms')
    write(raw / 'paired_benchmark.json', compact)
    write(raw / 'server_artifacts.json', {'server_only': server, 'annotation': artifact(annotation),
          'live_snapshot': artifact(args.live), 'baseline_summary': artifact(args.baseline),
          'category_replay': 'All 12 full-val metrics reproduced with absolute tolerance 1e-12 for all five models.',
          'gt_scale_bins': 'Disjoint original-image annotation area: small<32^2, medium<96^2, else large; all-area formal matches at maxDets100, not COCO area-specific matching.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', type=Path, default=Path('reports/5090/fresh_budgets/experiments.json'))
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    collect(parser.parse_args())
