"""Author FP32 fused-validator export, plus exact removal gate and latency."""
from common import *
import argparse,math,time
import torch
from ds_model import AuxModel,strip_auxiliary
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    verify_pin();env=environment();torch.set_num_threads(8)
    out=a.output.resolve();assert not out.exists(),f'Will not overwrite {out}';out.mkdir(parents=True)
    c=torch.load(a.checkpoint,map_location='cpu',weights_only=False);m=(c.get('ema') or c['model']).float().eval()
    weights=a.checkpoint
    if isinstance(m,AuxModel):
        native=strip_auxiliary(m)
        with torch.no_grad():
            torch.manual_seed(7);x=torch.rand(2,3,128,128)
            torch.testing.assert_close(m(x)[0],native(x)[0],rtol=0,atol=0)
        assert not any(k.startswith('aux_head.') for k in native.state_dict())
        weights=out/'native.pt'
        torch.save(dict(model=native,ema=None,epoch=c['epoch'],train_args=c['train_args']),weights)
    from ultralytics.utils import callbacks
    callbacks.add_integration_callbacks=lambda instance:None
    gt_path=DATA/'annotations/VisDrone2019-DET_val_coco.json'
    assert sha(gt_path)=='6f57aa4b42d64ee7b99b8a6a5b159ad19cba2eacf00bd86f23314966614bf4d8'
    gt=json.loads(gt_path.read_text());ids={Path(x['file_name']).stem:x['id'] for x in gt['images']}
    wanted={Path(x).stem for x in (DATA/'val_smoke.txt').read_text().splitlines()} if a.smoke else set(ids)
    class Validator(RTDETRValidator):
        def pred_to_json(self,predn,filename):
            for x1,y1,x2,y2,score,cls in predn.detach().cpu().tolist():
                self.jdict.append(dict(image_id=ids[Path(filename).stem],category_id=int(cls)+1,bbox=[x1,y1,x2-x1,y2-y1],score=score))
    model=RTDETR(str(weights))
    metrics=model.val(validator=Validator,data=str(DATA/('smoke.yaml' if a.smoke else 'data.yaml')),device='0',batch=4,imgsz=640,half=False,workers=2,save_json=True,plots=False,project=str(out.parent),name=out.name,exist_ok=True,conf=.001,max_det=300)
    pred_path=out/'predictions.json';boxes=json.loads(pred_path.read_text())
    assert {x['image_id'] for x in boxes}=={ids[k] for k in wanted}
    assert len(boxes)==300*len(wanted)
    assert all(math.isfinite(v) for x in boxes for v in x['bbox']+[x['score']])
    meta=dict(checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),checkpoint_epoch=int(c['epoch'])+1,deployment_checkpoint=str(weights),deployment_sha256=sha(weights),exact_aux_removal_verified=isinstance(m,AuxModel),gt_path=str(gt_path),gt_sha256=sha(gt_path),prediction_sha256=sha(pred_path),images=len(wanted),detections=len(boxes),native_metrics=metrics.results_dict,protocol=dict(engine='4090 author vendored source',precision='FP32 reloaded from FP16 EMA',preprocess='640x640 stretch RGB /255',batch=4,model_fusion='author AutoBackend default fusion',queries=300,confidence_filter=False,nms=False,coordinates='original image xywh; unrounded float',category_mapping='0..9 to COCO1..10',maxDets=[1,10,100],gt_sha256=sha(gt_path)),environment=env)
    # Separate, unobserved GPU-only FP32 inference timing on the author fused graph.
    gpu=model.model.cuda().float().eval();x=torch.rand(1,3,640,640,device='cuda')
    with torch.no_grad():
        for _ in range(20):gpu(x)
        torch.cuda.synchronize();timings=[]
        for _ in range(100):
            s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            s.record();gpu(x);e.record();torch.cuda.synchronize();timings.append(s.elapsed_time(e))
    import numpy as np
    meta['gpu_forward_fp32_batch1']=dict(warmup=20,iterations=100,mean_ms=float(np.mean(timings)),p50_ms=float(np.percentile(timings,50)),p95_ms=float(np.percentile(timings,95)),parameters=sum(p.numel() for p in gpu.parameters()),scope='5090 pure forward; excludes transfer/preprocess/postprocess; not deployment FP16')
    write(out/'prediction_metadata.json',meta)
    print(json.dumps({k:meta[k] for k in ['checkpoint_epoch','images','detections','exact_aux_removal_verified']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
