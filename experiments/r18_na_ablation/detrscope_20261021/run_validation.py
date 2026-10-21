"""Real checkpoint/CUDA/VisDrone validation, preserving raw evidence under runs/.

Run from the repository root with PYTHONPATH=$PWD:<detrscope>/src.
Only explicitly supplied trusted experiment checkpoints are unpickled.
"""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from detrscope import Debugger, AnalysisConfig
from detrscope.adapters.sodetr_data import CocoTargets, batch_from_ultralytics
from detrscope.core.identity import assert_outputs_equal, snapshot_output
from detrscope.core.io import read_json, write_json, sha256_file, state_sha256
from detrscope.core.records import iter_batches
from detrscope.evaluation.coco import CocoEvaluator, compare_gt
from detrscope.interventions import InterventionSpec
from source_adapter import PinnedR18NAAdapter


def git(repo, *args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


def hook_inventory(model):
    return {n: {k: len(getattr(m,k,{})) for k in ('_forward_hooks','_forward_pre_hooks','_backward_hooks')}
            for n,m in model.named_modules()}


def batches(paths, coco, device, batch_size):
    for start in range(0,len(paths),batch_size):
        part=paths[start:start+batch_size]
        images=[cv2.imread(str(p)) for p in part]
        if any(x is None for x in images):
            raise ValueError('Unreadable image')
        arrays=np.stack([cv2.resize(x,(640,640),interpolation=cv2.INTER_LINEAR)[:,:,::-1].transpose(2,0,1) for x in images])
        inputs=torch.from_numpy(np.ascontiguousarray(arrays)).to(device).float()/255
        yield batch_from_ultralytics({'im_file':part,'ori_shape':[im.shape[:2] for im in images]},inputs=inputs,coco=coco)


def native_boundary_check(model, batch, paths):
    # Check both native RTDETRDataset and Predictor preprocessing against our
    # lazy reader, then invoke the real native Predictor postprocess directly.
    from ultralytics.models.rtdetr.val import RTDETRDataset
    from ultralytics.models.rtdetr.predict import RTDETRPredictor
    from ultralytics.utils import ops
    dataset=RTDETRDataset(img_path=str(paths[0].parent),imgsz=640,batch_size=len(paths),augment=False,
                         hyp=SimpleNamespace(mask_ratio=4,overlap_mask=True),rect=False,cache=False,
                         data={'names':model.names,'nc':10})
    for i,p in enumerate(paths):
        j=dataset.im_files.index(str(p))
        native=dataset[j]['img'].float().to(batch.inputs.device)/255
        assert torch.equal(native,batch.inputs[i]), 'Native validation preprocessing differs'
    predictor=RTDETRPredictor(overrides={'imgsz':640,'conf':0.001,'max_det':300,'device':'0','verbose':False})
    predictor.imgsz=[640,640]
    originals=[cv2.imread(str(p)) for p in paths]
    transformed=predictor.pre_transform(originals)
    check=torch.from_numpy(np.ascontiguousarray(np.stack(transformed)[...,::-1].transpose(0,3,1,2))).to(batch.inputs.device).float()/255
    assert torch.equal(check,batch.inputs), 'Native prediction preprocessing differs'
    predictor.model=SimpleNamespace(names=model.names)
    predictor.batch=([str(p) for p in paths],originals,['']*len(paths))
    with torch.inference_mode():
        output=model(batch.inputs)
        native=predictor.postprocess(output,batch.inputs,originals)
    for result,image in zip(native,output[0]):
        box,prob=image[:,:4],image[:,4:]
        score,cls=prob.max(-1)
        keep=torch.where(score>0.001)[0]
        keep=keep[score[keep].argsort(descending=True)][:300]
        xyxy=ops.xywh2xyxy(box[keep])
        h,w=result.orig_shape
        xyxy[:,[0,2]]*=w; xyxy[:,[1,3]]*=h
        expected=torch.cat((xyxy,score[keep,None],cls[keep,None]),dim=1)
        assert torch.equal(result.boxes.data,expected), 'Native postprocessing differs'
    return {'native_validation_preprocessing_exact':True,'native_predictor_preprocessing_exact':True,
            'native_predictor_postprocessing_exact':True,'checked_images':len(paths),'fusion':False,'dtype':'float32'}


def predictions_from_raw(root, out):
    # Reproduce native per-query argmax, threshold, descending sort, max_det.
    # Do GPU arithmetic to retain exact FP32 conversion semantics; validate the
    # same implementation against the real native Predictor above.
    from ultralytics.utils.ops import xywh2xyxy
    predictions=[]; ids=[]
    for payload in iter_batches(root):
        for image in payload['images']:
            final=[s for s in image['stages'] if s['source']=='decoder'][-1]
            boxes=final['boxes'].cuda(); prob=final['probabilities'].cuda()
            score,cls=prob.max(-1); keep=torch.where(score>0.001)[0]
            keep=keep[score[keep].argsort(descending=True)][:300]
            xyxy=xywh2xyxy(boxes[keep]); h,w=image['metadata']['original_size']
            xyxy[:,[0,2]]*=w; xyxy[:,[1,3]]*=h
            xywh=xyxy.clone(); xywh[:,2:]=xyxy[:,2:]-xyxy[:,:2]
            iid=image['metadata']['image_id'];ids.append(iid)
            for b,s,c in zip(xywh.cpu().tolist(),score[keep].cpu().tolist(),cls[keep].cpu().tolist()):
                predictions.append({'image_id':iid,'category_id':int(c)+1,'bbox':b,'score':s})
    write_json(out,predictions)
    return predictions,ids


def collect(args):
    repo=Path(args.repo).resolve(); dest=Path(args.raw).resolve(); tracked=Path(args.tracked).resolve()
    tracked.mkdir(parents=True,exist_ok=True); dest.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0); cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0'); assert torch.cuda.is_available()
    checkpoint=Path(args.checkpoint).resolve()
    ck=torch.load(checkpoint,map_location='cpu')
    model=(ck.get('ema') or ck['model']).float().to(device).eval()
    before=state_sha256(model); hooks=hook_inventory(model)
    adapter=PinnedR18NAAdapter(model,repo=repo,pin=tracked/'source_pin.json')
    coco=CocoTargets(args.gt,model.names)
    data=read_json(args.gt); imdir=Path(args.images).resolve()
    paths=[imdir/Path(im['file_name']).name for im in sorted(data['images'],key=lambda x:x['id'])]
    assert len(paths)==548 and len(set(paths))==548
    if args.limit: paths=paths[:args.limit]
    sample=paths[:min(8,len(paths))]
    first=next(batches(sample[:2],coco,device,2))
    boundary=native_boundary_check(model,first,sample[:2])
    init=repo/'runs/detect/r18_na_ablation/rtdetr-r18-r1-150e-s0/initialization-before-resume-e36.json'
    run=checkpoint.parent.parent
    health=[json.loads(x) for x in (run/'health.jsonl').read_text().splitlines()]
    provenance={'source_commit':git(repo,'rev-parse','HEAD'),'checkpoint':str(checkpoint),'checkpoint_sha256':sha256_file(checkpoint),
                'gt_sha256':sha256_file(args.gt),'sdk_commit':git(args.sdk,'rev-parse','HEAD'),
                'sdk_dirty':bool(git(args.sdk,'status','--porcelain')),'training_epochs_completed':150,
                'runtime_scope':'Existing UAV_CV torch 2.1.2; source execution below SDK declared torch>=2.6 packaging target',
                'boundary_checks':boundary}
    print('BOUNDARY PASSED',boundary,flush=True)
    for name,selection,spec,probes,features in (
        ('baseline',paths,None,None,()),
        ('probe_sample',sample,None,'auto',('model.19','model.23','model.24.dec_bbox_head.2.layers.1')),
        ('identity_sample',sample,InterventionSpec(target='bbox_delta',layer_id=3,scale=1.0),None,()),
        ('zero_d3_sample',sample,InterventionSpec(target='bbox_delta',layer_id=3,scale=0.0),None,()),
    ):
        print('BEGIN',name,len(selection),datetime.now(timezone.utc).isoformat(),flush=True)
        dbg=Debugger(model,adapter,dest/name)
        kwargs=dict(provenance=provenance,probes=probes,features=features)
        loader=batches(selection,coco,device,args.batch)
        result=dbg.intervene(loader,spec,**kwargs) if spec else dbg.observe(loader,**kwargs)
        assert hook_inventory(model)==hooks, 'Hook leak'
        assert state_sha256(model)==before, 'State changed'
        print('COMPLETE',name,result['image_count'],flush=True)
    compatibility={'provenance':provenance,'boundary_checks':boundary,'hook_inventory_restored':True,
                   'model_state_unchanged':True,'model_state_sha256':before,'runtime':{'python':sys.version,'torch':str(torch.__version__),
                   'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'conda_prefix':os.environ.get('CONDA_PREFIX'),
                   'python_executable':sys.executable},'checkpoint_metadata':{k:ck.get(k) for k in ('epoch','date','version')},
                   'checkpoint_selection':'best.pt and last.pt are byte-identical; training-stripped EMA model, not raw final optimizer weights',
                   'checkpoint_gamma':{n:float(p.detach()) for n,p in model.named_parameters() if 'gamma' in n},
                   'training_last_health':health[-1],'initialization':read_json(init),'image_count':len(paths),'gt_count':len(data['annotations']),
                   'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'adapter_capabilities':adapter.capability_report(),
                   'raw_output':str(dest),'created_utc':datetime.now(timezone.utc).isoformat()}
    for name in ('baseline','probe_sample','identity_sample','zero_d3_sample'):
        m=read_json(dest/name/'manifest.json')
        compatibility.setdefault('conditions',{})[name]={'images':m['image_count'],'batches':len(m['batches']),
            'status':m['status'],'identity_verification':m['identity_verification'],
            'identity_passed_all':all(b['identity_passed'] and b['restoration_passed'] for b in m['batches']),
            'identity_intervention_passed_all':all(b.get('identity_intervention_passed',True) for b in m['batches']),
            'manifest_sha256':sha256_file(dest/name/'manifest.json')}
    write_json(tracked/'compatibility.json',compatibility)
    print('COLLECTION COMPLETE',flush=True)


def analyze_or_resume(root):
    manifest=root/'analysis/manifest.json'
    if manifest.exists():
        record=read_json(manifest)
        assert record['status']=='completed'
        assert record['record_manifest_sha256']==sha256_file(root/'manifest.json')
        for name,digest in record['artifacts'].items():
            assert sha256_file(root/'analysis'/name)==digest
        return read_json(root/'analysis/summary.json')
    if (root/'analysis').exists():
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        (root/'analysis').rename(root/('analysis_incomplete_'+stamp))
    return Debugger(output_dir=root).analyze(AnalysisConfig())


def offline(args):
    dest=Path(args.raw).resolve(); tracked=Path(args.tracked).resolve()
    torch.set_num_threads(4)
    evaluator=CocoEvaluator(args.gt)
    all_coco={}
    for name in ('baseline','probe_sample','identity_sample','zero_d3_sample'):
        root=dest/name
        print('ANALYZE',name,flush=True)
        dbg=Debugger(output_dir=root)
        summary=analyze_or_resume(root)
        dbg.report()
        predictions,ids=predictions_from_raw(root,root/'predictions.json')
        result=json.loads(json.dumps(evaluator.evaluate(predictions,image_ids=ids), default=lambda v: v.item() if isinstance(v,np.generic) else v))
        write_json(root/'coco.json',result); all_coco[name]=result
        write_json(tracked/f'{name}_summary.json',{k:v for k,v in summary.items() if k!='images'})
        shutil.copy2(root/'analysis/transition_stats.csv',tracked/f'{name}_transition_stats.csv')
        write_json(tracked/f'{name}_coco.json',{k:v for k,v in result.items() if k!='gt_states'})
        if name=='probe_sample':
            shutil.copy2(root/'analysis/probes.csv',tracked/'probe_statistics.csv')
        print('COCO',name,result['metrics'],flush=True)
    # Full raw-record replay in a second directory; hard links preserve evidence
    # without doubling tensor storage. Analysis outputs must match byte for byte.
    replay=dest/'offline_replay'; replay.mkdir(exist_ok=True)
    shutil.copy2(dest/'baseline/manifest.json',replay/'manifest.json')
    if not (replay/'raw').exists():
        shutil.copytree(dest/'baseline/raw',replay/'raw',copy_function=os.link)
    analyze_or_resume(replay); Debugger(output_dir=replay).report()
    files=lambda p:{str(f.relative_to(p)):sha256_file(f) for f in p.rglob('*') if f.is_file()}
    a=files(dest/'baseline/analysis'); b=files(replay/'analysis')
    assert a==b, 'Offline analysis replay differs'
    pred_replay,ids=predictions_from_raw(replay,replay/'predictions.json')
    replay_coco=json.loads(json.dumps(evaluator.evaluate(pred_replay,image_ids=ids), default=lambda v: v.item() if isinstance(v,np.generic) else v))
    assert replay_coco==all_coco['baseline'], 'Offline COCO replay differs'
    replay_check={'full_dataset_images':len(ids),'analysis_artifacts_compared':len(a),'all_analysis_sha256_equal':True,
                  'coco_result_exactly_equal':True,'predictions_json_sha256_equal':sha256_file(replay/'predictions.json')==sha256_file(dest/'baseline/predictions.json')}
    write_json(tracked/'offline_replay.json',replay_check)
    assert all_coco['probe_sample']==all_coco['identity_sample'], 'Identity intervention changed COCO results'
    comparison=Debugger(output_dir=dest/'probe_sample').compare(dest/'zero_d3_sample',dest/'zero_d3_comparison',
                     baseline_coco=dest/'probe_sample/coco.json',candidate_coco=dest/'zero_d3_sample/coco.json')
    write_json(tracked/'zero_d3_formal_gt_changes.json',compare_gt(all_coco['probe_sample'],all_coco['zero_d3_sample']))
    for path in sorted((dest/'zero_d3_comparison').rglob('*')):
        if path.is_file() and path.suffix in ('.json','.csv','.md') and path.stat().st_size<2_000_000:
            shutil.copy2(path,tracked/('comparison_'+path.name))
    write_json(tracked/'completion.json',{'status':'completed','offline_replay':replay_check,'finished_utc':datetime.now(timezone.utc).isoformat()})
    print('ALL VALIDATION COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['collect','offline'])
    p.add_argument('--repo',default=str(Path(__file__).resolve().parents[3]))
    p.add_argument('--sdk',default='/home/zjnu/文档/ZHJ_WorkSpace/detrscope')
    p.add_argument('--tracked',default=str(Path(__file__).resolve().parent))
    p.add_argument('--checkpoint',required=True);p.add_argument('--gt',required=True);p.add_argument('--images',required=True)
    p.add_argument('--raw',required=True);p.add_argument('--batch',type=int,default=2);p.add_argument('--limit',type=int)
    args=p.parse_args(); (collect if args.phase=='collect' else offline)(args)

if __name__=='__main__': main()
