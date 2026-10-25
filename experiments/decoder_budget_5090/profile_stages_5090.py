"""Within-forward CUDA stage timing; no model surgery or training."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import math
import random
import statistics
import subprocess

import torch

from experiments.decoder_budget_5090.run import configure, batches, query_budget, environment
from detrscope.adapters.sodetr_data import CocoTargets
from detrscope.core.identity import assert_outputs_equal, rng_state, assert_rng_equal
from detrscope.core.io import state_sha256

ROOT = Path('/home/ubuntu/文档/ZHJ_UAV/ultralytics')
RUNS = ROOT / 'runs/decoder_budget_5090/5090-c0c2-baselines-20261024-s0-r1'
DATA = ROOT.parent / 'SO_DETR/datasets/VisDrone2019-DET-YOLO'
OUT = ROOT / 'runs/diagnostics' / ('5090-stage-latency-' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S'))
OUT.mkdir(parents=True, exist_ok=False)
configure()
STAGES = ['backbone_0_to_4', 'neck_encoder_5_to_23', 'input_projection_and_query_selection',
          'decoder_and_prediction_heads', 'output_assembly']
report = {'scope': 'Within the same native unfused FP32 batch1 forward, 640 square; CUDA stream elapsed includes dispatch-induced idle gaps, not pure kernel busy time; no pre/postprocess or transfers',
          'rounds': 5, 'warmup_each_block': 10, 'repetitions_each_block': 50,
          'environment': environment('cuda:0'), 'models': {}}


def clocks():
    p = subprocess.run(['nvidia-smi', '--query-gpu=clocks.current.sm,clocks.current.memory,pstate,temperature.gpu',
                        '--format=csv,noheader'], capture_output=True, text=True)
    return p.stdout.strip()


def stats(values):
    return {'mean_ms': math.fsum(values) / len(values), 'median_ms': statistics.median(values),
            'min_ms': min(values), 'max_ms': max(values), 'samples': len(values)}


with torch.inference_mode():
    for variant in ('c0', 'c2'):
        path = RUNS / f'5090-rtdetr-r18-{variant}-150e-s0/weights/best.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        model = (saved.get('ema') or saved['model']).float().eval().cuda()
        head = model.model[-1]
        before = state_sha256(model)
        coco = CocoTargets(DATA / 'annotations/VisDrone2019-DET_val_coco.json', model.names)
        first = sorted(coco.images.values(), key=lambda r: r['id'])[0]
        x = next(batches([DATA / 'images/val' / Path(first['file_name']).name], coco, 'cuda:0', 1, 640)).inputs
        events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
        for event in events:
            event.record()
        torch.cuda.synchronize()
        for _ in range(150):
            model(x)
        torch.cuda.synchronize()
        model_report = {'checkpoint': str(path), 'blocks': [], 'identity_checks': {}}
        for round_index in range(5):
            order = [(q, mode) for q in (300, 200, 150) for mode in ('bare', 'staged')]
            random.Random(20261025 + round_index).shuffle(order)
            for budget, mode in order:
                with query_budget(head, budget):
                    native = model(x)
                    if native[0].shape[1] != budget:
                        raise RuntimeError('Query budget did not change native output')
                    initial_rng = rng_state()
                    handles = []
                    if mode == 'staged':
                        handles = [model.model[4].register_forward_hook(lambda m, a, y: events[1].record()),
                                   model.model[23].register_forward_hook(lambda m, a, y: events[2].record()),
                                   head.decoder.register_forward_pre_hook(lambda m, a: events[3].record()),
                                   head.decoder.register_forward_hook(lambda m, a, y: events[4].record())]
                    try:
                        observed = model(x)
                        assert_outputs_equal(native, observed)
                        assert_rng_equal(initial_rng, rng_state())
                        model_report['identity_checks'][str(budget)] = True
                        for _ in range(10):
                            model(x)
                        torch.cuda.synchronize()
                        values = {'whole': []}
                        if mode == 'staged':
                            values.update({stage: [] for stage in STAGES})
                        for _ in range(50):
                            events[0].record()
                            model(x)
                            events[5].record()
                            events[5].synchronize()
                            whole = events[0].elapsed_time(events[5])
                            values['whole'].append(whole)
                            if mode == 'staged':
                                parts = [events[i].elapsed_time(events[i+1]) for i in range(5)]
                                if abs(sum(parts) - whole) > .02:
                                    raise RuntimeError('Stage accounting does not match native forward interval')
                                for stage, duration in zip(STAGES, parts):
                                    values[stage].append(duration)
                    finally:
                        for handle in reversed(handles):
                            handle.remove()
                    assert_outputs_equal(native, model(x))
                    model_report['blocks'].append({'round': round_index + 1, 'queries': budget, 'mode': mode,
                                                   'measurements': values, 'clocks_after_block': clocks()})
            report['models'][variant] = model_report
            (OUT / 'assessment.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print(variant, 'round', round_index + 1, 'completed', flush=True)
        summary = {}
        for budget in (300, 200, 150):
            selected = [b for b in model_report['blocks'] if b['queries'] == budget]
            bare = [v for b in selected if b['mode'] == 'bare' for v in b['measurements']['whole']]
            staged = [b for b in selected if b['mode'] == 'staged']
            instrumented = [v for b in staged for v in b['measurements']['whole']]
            total = stats(instrumented)['mean_ms']
            summary[str(budget)] = {'bare_whole': stats(bare), 'staged_whole': stats(instrumented),
                                    'instrumentation_overhead_percent': 100*(total/stats(bare)['mean_ms']-1), 'stages': {}}
            for stage in STAGES:
                values = [v for b in staged for v in b['measurements'][stage]]
                entry = stats(values)
                entry['fraction_of_staged_forward'] = entry['mean_ms'] / total
                means = [stats(b['measurements'][stage])['mean_ms'] for b in staged]
                entry['round_mean_min_ms'], entry['round_mean_max_ms'] = min(means), max(means)
                summary[str(budget)]['stages'][stage] = entry
        model_report['summary'] = summary
        if head.num_queries != 300 or state_sha256(model) != before:
            raise RuntimeError('Model state or query budget changed')
        if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
            raise RuntimeError('Profiling hooks leaked')
        model_report.update(model_state_unchanged=True, query_budget_restored=True, hooks_remaining=0)
        report['models'][variant] = model_report
        (OUT / 'assessment.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(variant, json.dumps(summary, ensure_ascii=False), flush=True)
        del native, observed, model, head, saved
        torch.cuda.empty_cache()

(OUT / 'profile_5090.py').write_text(Path(__file__).read_text())
print('OUTPUT', OUT, flush=True)
