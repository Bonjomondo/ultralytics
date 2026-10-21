"""Independent native validator on the same saved checkpoint and all 548 images."""
from pathlib import Path
import json
import torch
from ultralytics import RTDETR
from detrscope.core.io import write_json, sha256_file

repo=Path(__file__).resolve().parents[3]
weights=repo/'runs/detect/r18_na_ablation/rtdetr-r18-r1-150e-s0/weights/best.pt'
model=RTDETR(str(weights))
result=model.val(data=str(repo/'configs/visdrone-local.yaml'),device='0',imgsz=640,batch=2,workers=0,
                 half=False,plots=False,verbose=False,save_json=True,
                 project=str(repo/'runs/detect/r18_na_ablation'),name='detrscope_native_val_20261021',exist_ok=False)
write_json(Path(__file__).with_name('native_val_internal_metrics.json'),{'metrics':result.results_dict,
    'checkpoint_sha256':sha256_file(weights),'note':'Unmodified native Ultralytics Validator internal metrics, distinct from pycocotools COCOeval',
    'save_dir':str(result.save_dir),'torch':str(torch.__version__),'batch':2,'imgsz':640,'half':False})
print('NATIVE VALIDATOR',result.results_dict)

# Independently run formal COCOeval on the native validator's saved detections.
from detrscope.core.io import read_json
from detrscope.evaluation.coco import CocoEvaluator
import numpy as np
annotation=Path('/home/zjnu/文档/ZHJ_WorkSpace/datasets/VisDrone2019-DET-YOLO/annotations/VisDrone2019-DET_val_coco.json')
images={Path(im['file_name']).name:im['id'] for im in read_json(annotation)['images']}
native_predictions=read_json(Path(result.save_dir)/'predictions.json')
for row in native_predictions:
    row['image_id']=images[row['file_name']]
formal=CocoEvaluator(annotation).evaluate(native_predictions)
portable=json.loads(json.dumps(formal,default=lambda x:x.item() if isinstance(x,np.generic) else x))
write_json(Path(__file__).with_name('native_val_formal_coco.json'),{k:v for k,v in portable.items() if k!='gt_states'})
base=read_json(Path(__file__).with_name('formal_evaluation.json'))
write_json(Path(__file__).with_name('native_vs_recorded_coco.json'),{'native_minus_recorded_pp':{k:(portable['metrics'][k]-v)*100 for k,v in base['metrics'].items()},
    'scope':'Same checkpoint, images and GT. Native fused Validator JSON rounds boxes to 3 decimals and scores to 5 decimals; recorded unfused Predictor-compatible export retains full FP32 precision.',
    'native_prediction_file':str(Path(result.save_dir)/'predictions.json')})
print('NATIVE FORMAL',portable['metrics'])
