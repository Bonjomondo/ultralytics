"""Remeasure the trained baseline and four fresh budgets in randomized paired rounds.

Reuse the exact checkpoint-linked engines from completed evaluations; never train
or change a model. Raw timings and engine/checkpoint hashes are written server-side.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import random

import torch

from experiments.decoder_budget_5090.profile_runtime_5090 import (
    DEFAULT_DATA, configure, environment, load_input, measure, prediction, sha, stats, write,
)


def read(path):
    return json.loads(path.read_text())


def engine_function(metadata, inputs):
    import tensorrt as trt
    artifact = next(a for a in metadata['server_artifacts'] if a['path'].endswith('.engine'))
    path = Path(artifact['path'])
    if sha(path) != artifact['sha256']:
        raise RuntimeError('Engine hash mismatch: ' + str(path))
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    context = engine.create_execution_context()
    assert tuple(engine.get_tensor_shape('images')) == tuple(inputs.shape)
    assert engine.get_tensor_dtype('images') == trt.float32
    context.set_tensor_address('images', inputs.data_ptr())
    output = torch.empty(tuple(engine.get_tensor_shape('predictions')), device='cuda', dtype=torch.float32)
    context.set_tensor_address('predictions', output.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream
    assert stream != 0

    def run():
        # Keep the full TensorRT object chain alive until all measurements finish.
        assert logger is not None and runtime is not None and engine is not None
        if not context.execute_async_v3(stream):
            raise RuntimeError('TensorRT enqueue failed')
        return output

    return run


def run(args):
    configure()
    args.output.mkdir(parents=True, exist_ok=False)
    old = read(args.baseline)
    fresh = [read(p) for p in sorted(args.root.glob('*150e-s0-eval/summary.json'))]
    assert len(fresh) == 4 and all(x['status'] == 'completed' for x in fresh)
    baseline_group = old['performance_groups']['d3/model']
    entries = [('c2-q300-p4-d3', old['checkpoint'], baseline_group['backends'], args.baseline)]
    entries += [(d['variant'], d['checkpoint'], d['benchmark']['backends'],
                 args.root / (Path(d['checkpoint']['path']).parents[1].name + '-eval') / 'summary.json')
                for d in fresh]
    x, identity = load_input(DEFAULT_DATA)
    functions, conditions = {}, {}
    for variant, checkpoint, backends, receipt in entries:
        assert sha(Path(checkpoint['path'])) == checkpoint['sha256']
        saved = torch.load(checkpoint['path'], map_location='cpu', weights_only=False)
        model = (saved.get('ema') or saved['model']).float().eval().cuda()
        for backend in ('eager_fp32', 'trt_fp32', 'trt_fp16'):
            key = variant + '/' + backend
            fn = (lambda model=model: prediction(model(x))) if backend == 'eager_fp32' else engine_function(backends[backend]['engine'], x)
            output = fn()
            assert torch.isfinite(output).all() and output.shape[1] == model.model[-1].num_queries
            functions[key] = fn
            conditions[key] = {'variant': variant, 'backend': backend, 'checkpoint': checkpoint,
                               'source_receipt': {'path': str(receipt.resolve()), 'sha256': sha(receipt)},
                               'engine': backends[backend].get('engine'), 'blocks': []}
    result = {'schema': '5090.fresh-paired-benchmark.v1', 'status': 'running',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'environment': environment('cuda:0'),
              'source_sha256': {str(Path(__file__).resolve()): sha(Path(__file__))}, 'input': identity,
              'protocol': {'scope': 'GPU-resident batch1 640 model forward; no preprocessing, transfer or postprocessing',
                           'backends': ['eager_fp32', 'trt_fp32', 'trt_fp16'], 'rounds': args.rounds,
                           'samples_per_round': 50, 'warmup_per_condition': 150, 'seed': 20261026,
                           'weights': 'own trained best.pt; existing engines SHA verified; no rebuilt tactics',
                           'precision': 'FP16 benchmark only; AP independently evaluated in unfused native FP32'},
              'round_order': [], 'conditions': conditions}
    for fn in functions.values():
        for _ in range(150):
            fn()
    torch.cuda.synchronize()
    for round_index in range(args.rounds):
        order = list(functions)
        random.Random(20261026 + round_index).shuffle(order)
        result['round_order'].append(order)
        for key in order:
            block = measure(functions[key], 10, 50)
            block['round'] = round_index + 1
            conditions[key]['blocks'].append(block)
        write(args.output / 'progress.json', result)
        print('completed round', round_index + 1, flush=True)
    for condition in conditions.values():
        for clock in ('cuda', 'wall'):
            condition[clock + '_stats_ms'] = stats([v for b in condition['blocks'] for v in b[clock + '_ms']])
    result.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat())
    write(args.output / 'assessment.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=20)
    args = parser.parse_args()
    with open('/tmp/zhj-uav-5090-gpu0.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with torch.inference_mode():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run(args)
            stream.synchronize()
