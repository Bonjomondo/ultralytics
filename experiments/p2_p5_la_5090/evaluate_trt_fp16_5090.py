"""Full-val FP16 TensorRT AP for two trained engines; no training or timing."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import tensorrt as trt
from pycocotools.coco import COCO

from experiments.decoder_budget_5090.train_5090 import now, runtime_environment, sha256_file, write_atomic
from experiments.p2_p5_la_5090.evaluate import category_ids, coco_metrics, native_predictor
from experiments.p2_p5_la_5090.train import NAMES, configure


def evaluate(args):
    configure()
    benchmark = json.loads(args.benchmark.read_text())
    training = json.loads(args.training.read_text())
    native = json.loads(args.native_evaluation.read_text())
    if benchmark['status'] != 'completed' or native['scope'] != 'full_val':
        raise ValueError('Require completed trained-model timing and native full validation')
    gt = COCO(str(args.annotations))
    images = sorted(gt.dataset['images'], key=lambda item: item['id'])
    ids = [item['id'] for item in images]
    if len(ids) != 548 or ids != native['protocol']['image_ids']:
        raise ValueError('Validation image identity differs from native FP32 evaluation')
    annotation_sha = sha256_file(args.annotations)
    if annotation_sha != native['protocol']['annotations_sha256']:
        raise ValueError('Annotation identity differs from native FP32 evaluation')
    categories = category_ids(gt, NAMES)
    args.output.mkdir(parents=True, exist_ok=False)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engines = {}
    for arm in ('c2', 'c2-la'):
        condition = benchmark['conditions'][arm + '/trt_fp16']
        if condition['status'] != 'completed' or not condition['engine']['fp16_allowed']:
            raise ValueError('A trained FP16 engine is unavailable: ' + arm)
        if arm == 'c2-la' and condition['checkpoint']['sha256'] != training['checkpoints']['best']['sha256']:
            raise ValueError('Joint engine and native evaluation use different checkpoints')
        engine_file = next(v for v in condition['engine']['server_artifacts'] if v['path'].endswith('.engine'))
        path = Path(engine_file['path'])
        if sha256_file(path) != engine_file['sha256']:
            raise ValueError('Engine hash differs from benchmark receipt')
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        context = engine.create_execution_context()
        if tuple(engine.get_tensor_shape('images')) != (1, 3, 640, 640) or engine.get_tensor_dtype('images') != trt.float32:
            raise ValueError('Unexpected FP16 engine input')
        if tuple(engine.get_tensor_shape('predictions')) != (1, 300, 14) or engine.get_tensor_dtype('predictions') != trt.float32:
            raise ValueError('Unexpected FP16 engine output')
        output = torch.empty((1, 300, 14), device='cuda', dtype=torch.float32)
        context.set_tensor_address('predictions', output.data_ptr())
        engines[arm] = {'engine': engine, 'context': context, 'output': output, 'artifact': engine_file,
                        'checkpoint': condition['checkpoint']}
    predictor = native_predictor(dict(enumerate(NAMES)))
    predictor.model = SimpleNamespace(names=dict(enumerate(NAMES)))
    record = {'schema': '5090.joint150.trt-fp16-coco.v1', 'status': 'running', 'started_utc': now(),
              'environment': runtime_environment(), 'tensorrt': trt.__version__,
              'protocol': {'scope': 'full_val', 'images': len(images), 'image_ids': ids, 'annotation_path': str(args.annotations.resolve()),
                           'annotation_sha256': annotation_sha, 'category_ids': categories, 'batch': 1, 'size': 640,
                           'engine_precision': 'TensorRT FP16 enabled; engine may choose FP32 layers; FP32 I/O',
                           'preprocess': 'native scale_fill RGB BCHW contiguous /255', 'conf': .001,
                           'prediction_max_det': 300, 'formal_max_dets': [1,10,100],
                           'postprocess': 'native per-query argmax, sorted confidence, no NMS/no JSON rounding'},
              'models': {arm: {'engine': e['artifact'], 'checkpoint': e['checkpoint']} for arm,e in engines.items()}}
    predictions = {arm: [] for arm in engines}
    stream = torch.cuda.current_stream()
    if stream.cuda_stream == 0:
        raise RuntimeError('Require a non-default shared stream for TensorRT')
    for index, meta in enumerate(images):
        path = args.images / Path(meta['file_name']).name
        original = cv2.imread(str(path))
        if original is None or original.shape[:2] != (meta['height'], meta['width']):
            raise ValueError('Validation image missing or has wrong dimensions: ' + str(path))
        array = np.stack(predictor.pre_transform([original]))[..., ::-1].copy()
        inputs = torch.from_numpy(array).permute(0,3,1,2).contiguous().cuda().float() / 255
        if index == 0:
            input_sha = hashlib.sha256(inputs.cpu().numpy().tobytes()).hexdigest()
            if input_sha != benchmark['input']['input_sha256']:
                raise ValueError('First input differs from the benchmark input')
            record['first_input_sha256'] = input_sha
        predictor.batch = ([str(path)], [original], [''])
        for arm, item in engines.items():
            item['context'].set_tensor_address('images', inputs.data_ptr())
            if not item['context'].execute_async_v3(stream.cuda_stream):
                raise RuntimeError('TensorRT enqueue failed for ' + arm)
            output = item['output'].clone()
            if not torch.isfinite(output).all():
                raise FloatingPointError('Nonfinite TensorRT output for ' + arm)
            result = predictor.postprocess((output,None),None,[original])[0]
            for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                x1,y1,x2,y2 = box
                predictions[arm].append({'image_id': meta['id'], 'category_id': categories[int(cls)],
                                         'bbox': [x1,y1,x2-x1,y2-y1], 'score': score})
        if (index+1) % 100 == 0:
            print('FP16 predictions completed:', index+1, '/', len(images), flush=True)
    for arm in ('c2', 'c2-la'):
        path = args.output / (arm + '-predictions.json')
        write_atomic(path,predictions[arm])
        metrics,per_class = coco_metrics(gt,predictions[arm],ids)
        record['models'][arm].update(metrics=metrics,per_class=per_class,prediction_count=len(predictions[arm]),
                                     predictions={'path': str(path.resolve()), 'bytes': path.stat().st_size,
                                                  'sha256': sha256_file(path)})
    record.update(status='completed',completed_utc=now())
    write_atomic(args.output / 'summary.json',record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--native-evaluation',type=Path,required=True)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with open('/tmp/zhj-uav-5090-gpu0.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with torch.inference_mode():
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                evaluate(args)
            stream.synchronize()
