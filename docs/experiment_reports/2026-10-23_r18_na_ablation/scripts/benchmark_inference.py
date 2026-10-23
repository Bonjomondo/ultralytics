"""Batch=1 native PyTorch benchmarks for five completed RT-DETR R18 ablations.

Run only on an idle GPU; no training processes are stopped by this script.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path('/home/zjnu/文档/ZHJ_WorkSpace/ultralytics')
RUNS = ROOT / 'runs/detect/r18_na_ablation'
RUN_NAMES = {
    'r0': 'rtdetr-r18-r0-150e-s0',
    'r1': 'rtdetr-r18-r1-150e-s0',
    'r2a': 'rtdetr-r18-r2a-150e-s0-20261021',
    'r2b': 'rtdetr-r18-r2b-150e-s0-20261021',
    'r2c': 'rtdetr-r18-r2c-150e-s0-20261021',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def gpu_status():
    return subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
        '--format=csv,noheader',
    ], text=True).strip()


def worker(out):
    other_compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
    if other_compute:
        raise RuntimeError(f'GPU already has compute processes: {other_compute}')
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import gc
    import random
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import RTDETR
    from ultralytics.models.rtdetr.predict import RTDETRPredictor

    allowed = sorted(os.sched_getaffinity(0))
    cores = [c for c in [24, 25, 26, 27] if c in allowed]
    if len(cores) != 4:
        cores = allowed[-4:]
    os.sched_setaffinity(0, cores)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    image_dir = ROOT.parent / 'datasets/VisDrone2019-DET-YOLO/VisDrone2019-DET-val/images'
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg', '.png', '.jpeg'})
    selected = [paths[int(i)] for i in np.linspace(0, len(paths) - 1, 64)]
    images = [cv2.imread(str(p)) for p in selected]
    assert all(im is not None for im in images)
    configs = [(v, p) for p in ['fp32', 'fp16'] for v in RUN_NAMES]
    random.Random(20261022).shuffle(configs)
    environment = {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                   'cudnn': torch.backends.cudnn.version(), 'ultralytics': ultralytics.__version__,
                   'gpu': gpu_status(), 'gpu_compute_processes_before_start': other_compute, 'cpu_affinity': cores, 'torch_threads': 4,
                   'tf32': False, 'cudnn_benchmark': True}
    protocol = {'batch': 1, 'input_shape': [1, 3, 640, 640], 'backend': 'PyTorch eager, fused, eval, inference_mode',
                'warmup_forwards': 50, 'rounds': 3, 'forward_iterations_per_round': 150,
                'pipeline_images_per_round': 64, 'pipeline_confidence': 0.25, 'max_det': 300,
                'pipeline': 'cached decoded BGR -> native RTDETR square scale-fill preprocessing -> H2D -> forward -> native postprocess -> boxes CPU -> CUDA synchronize',
                'excluded': ['disk/JPEG decoding', 'model loading', 'rendering', 'TensorRT/torch.compile', 'batched throughput'],
                'forward_wall': 'perf_counter with CUDA synchronization, input already resident on GPU',
                'forward_gpu': 'CUDA events, same forwards',
                'fps': '1000 / arithmetic mean latency in milliseconds, sequential batch=1',
                'memory': 'peak torch allocator allocated/reserved MiB in benchmark process; other processes and CUDA context memory are excluded',
                'configuration_order': configs, 'images': [str(p) for p in selected]}
    save(out / 'protocol.json', {'environment': environment, 'protocol': protocol})
    records = []

    def summarize(values):
        a = np.asarray(values, dtype=float)
        return {'mean_ms': float(a.mean()), 'median_ms': float(np.median(a)),
                'p95_ms': float(np.percentile(a, 95)), 'std_ms': float(a.std()),
                'fps': float(1000 / a.mean()), 'samples': len(values)}

    with torch.inference_mode():
        for variant, precision in configs:
            print(now(), 'Benchmarking', variant, precision, flush=True)
            checkpoint = RUNS / RUN_NAMES[variant] / 'weights/best.pt'
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            wrapper = RTDETR(str(checkpoint))
            parameter_count = sum(p.numel() for p in wrapper.model.parameters())
            predictor = RTDETRPredictor(overrides={'imgsz': 640, 'device': '0', 'half': precision == 'fp16',
                                                   'conf': 0.25, 'max_det': 300, 'verbose': False, 'save': False})
            predictor.setup_model(wrapper.model, verbose=False)
            predictor.imgsz = [640, 640]
            model = predictor.model.model
            model.eval()
            fused_count = sum(p.numel() for p in model.parameters())
            dtype = torch.float16 if precision == 'fp16' else torch.float32
            assert next(model.parameters()).dtype == dtype
            assert predictor.model.fp16 == (precision == 'fp16')
            x = torch.rand((1, 3, 640, 640), device='cuda', dtype=dtype)
            for _ in range(50):
                predictions = model(x)
                del predictions
            torch.cuda.synchronize()
            predictions = model(x)
            assert torch.isfinite(predictions[0]).all().item(), f'Non-finite output: {variant} {precision}'
            del predictions
            forward_wall, forward_gpu, pipeline_wall, round_stats = [], [], [], []
            gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.reset_peak_memory_stats()
            for round_index in range(3):
                this_wall, this_gpu = [], []
                for _ in range(150):
                    start = time.perf_counter()
                    gpu_start.record()
                    predictions = model(x)
                    gpu_end.record()
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    this_wall.append(elapsed)
                    this_gpu.append(gpu_start.elapsed_time(gpu_end))
                    del predictions
                forward_wall.extend(this_wall)
                forward_gpu.extend(this_gpu)
                round_stats.append({'round': round_index + 1, 'forward_wall': summarize(this_wall), 'forward_gpu': summarize(this_gpu)})
            forward_allocated = torch.cuda.max_memory_allocated() / 2**20
            forward_reserved = torch.cuda.max_memory_reserved() / 2**20

            def pipeline(index):
                original = images[index]
                predictor.batch = ([str(selected[index])], [original], [''])
                tensor = predictor.preprocess([original])
                predictions = model(tensor)
                results = predictor.postprocess(predictions, tensor, [original])
                boxes_cpu = results[0].boxes.data.cpu()
                torch.cuda.synchronize()
                return int(boxes_cpu.shape[0])

            for index in range(5):
                pipeline(index)
            torch.cuda.reset_peak_memory_stats()
            detection_counts = []
            for round_index in range(3):
                this_pipeline = []
                for index in range(len(images)):
                    start = time.perf_counter()
                    detections = pipeline(index)
                    this_pipeline.append((time.perf_counter() - start) * 1000)
                    if round_index == 0:
                        detection_counts.append(detections)
                pipeline_wall.extend(this_pipeline)
                round_stats[round_index]['pipeline_wall'] = summarize(this_pipeline)
            record = {'variant': variant, 'precision': precision, 'checkpoint': str(checkpoint), 'checkpoint_sha256': digest,
                      'parameters_before_fusion': parameter_count, 'parameters_fused': fused_count,
                      'runtime_parameter_dtype': str(next(model.parameters()).dtype), 'finite_output_check_passed': True,
                      'forward_wall': summarize(forward_wall), 'forward_gpu': summarize(forward_gpu),
                      'pipeline_wall': summarize(pipeline_wall), 'rounds': round_stats,
                      'forward_peak_allocated_mib': forward_allocated, 'forward_peak_reserved_mib': forward_reserved,
                      'pipeline_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                      'pipeline_peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
                      'raw_timings_ms': {'forward_wall': forward_wall, 'forward_gpu': forward_gpu, 'pipeline_wall': pipeline_wall},
                      'detection_counts_first_round': detection_counts, 'gpu_after_configuration': gpu_status()}
            records.append(record)
            save(out / 'benchmark.json', {'environment': environment, 'protocol': protocol, 'records': records})
            print(now(), variant, precision, 'forward', record['forward_wall'], 'pipeline', record['pipeline_wall'], flush=True)
            del predictor, model, wrapper, x
            gc.collect()
            torch.cuda.empty_cache()
    fields = ['variant', 'precision', 'parameters', 'forward_mean_ms', 'forward_median_ms', 'forward_p95_ms',
              'forward_fps', 'gpu_mean_ms', 'pipeline_mean_ms', 'pipeline_p95_ms', 'pipeline_fps',
              'forward_peak_allocated_mib', 'pipeline_peak_allocated_mib']
    with (out / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in sorted(records, key=lambda r: (r['precision'], r['variant'])):
            writer.writerow({'variant': r['variant'], 'precision': r['precision'], 'parameters': r['parameters_before_fusion'],
                             **{f'forward_{k}': r['forward_wall'][k] for k in ['mean_ms', 'median_ms', 'p95_ms', 'fps']},
                             'gpu_mean_ms': r['forward_gpu']['mean_ms'],
                             **{f'pipeline_{k}': r['pipeline_wall'][k] for k in ['mean_ms', 'p95_ms', 'fps']},
                             'forward_peak_allocated_mib': r['forward_peak_allocated_mib'],
                             'pipeline_peak_allocated_mib': r['pipeline_peak_allocated_mib']})
    print(now(), 'BENCHMARK_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    worker(args.output)
