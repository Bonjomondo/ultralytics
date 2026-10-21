"""Export a small, Git-readable evidence package; no weights/images in Git."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import csv
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
from detrscope.core.io import read_json, write_json, sha256_file
from detrscope.core.records import iter_batches
from detrscope.evaluation.coco import CocoEvaluator
from run_validation import predictions_from_raw


def table(path,rows,fields=None):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--raw');a=p.parse_args();repo=Path(a.repo).resolve()
    tracked=Path(__file__).resolve().parent;train=repo/'runs/detect/r18_na_ablation/rtdetr-r18-r1-150e-s0'
    raw=Path(a.raw).resolve() if a.raw else train/'detrscope_20261021';gt=Path('/home/zjnu/文档/ZHJ_WorkSpace/datasets/VisDrone2019-DET-YOLO/annotations/VisDrone2019-DET_val_coco.json')
    torch.set_num_threads(2)
    for src,dst in [('results.csv','training_metrics.csv'),('health.jsonl','training_health.jsonl'),
                    ('args.yaml','training_args.yaml'),('initialization.json','initialization_resume.json'),
                    ('initialization-before-resume-e36.json','initialization_original.json')]:
        shutil.copy2(train/src,tracked/dst)
    preds,ids=predictions_from_raw(raw/'baseline',raw/'formal_predictions.json')
    result=json.loads(json.dumps(CocoEvaluator(gt).evaluate(preds,image_ids=ids),default=lambda x:x.item()))
    write_json(tracked/'formal_evaluation.json',{k:v for k,v in result.items() if k!='gt_states'})
    # Native COCOeval precision tensor, independently extract class/size rows.
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with redirect_stdout(io.StringIO()):
        c=COCO(str(gt));d=c.loadRes(preds);e=COCOeval(c,d,'bbox');e.params.imgIds=ids
        e.evaluate();e.accumulate();e.summarize()
    for key,value in zip(('AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl'),e.stats):
        assert result['metrics'][key]==value
    precision=e.eval['precision']
    def avg(v):
        valid=v[v>-1];return float(valid.mean()) if valid.size else None
    counts=Counter(x['category_id'] for x in c.dataset['annotations'])
    rows=[]
    for k,cid in enumerate(e.params.catIds):
        rows.append({'category_id':int(cid),'class_name':c.cats[cid]['name'],'gt_count':counts[cid],
                     'AP':avg(precision[:,:,k,0,-1]),'AP50':avg(precision[0,:,k,0,-1]),
                     'AP75':avg(precision[5,:,k,0,-1]),'APs':avg(precision[:,:,k,1,-1]),
                     'APm':avg(precision[:,:,k,2,-1]),'APl':avg(precision[:,:,k,3,-1])})
    table(tracked/'per_class_coco.csv',rows)
    # Sensitivity to COCO maxDets; keep the standard 100 protocol separate.
    with redirect_stdout(io.StringIO()):
        e300=COCOeval(c,d,'bbox');e300.params.imgIds=ids;e300.params.maxDets=[1,10,300]
        e300.evaluate();e300.accumulate()
    pr=e300.eval['precision']
    write_json(tracked/'maxdets_sensitivity.json',{'standard_maxdets100':result['metrics'],
        'nonstandard_maxdets300':{'AP':avg(pr[:,:,:,0,-1]),'AP50':avg(pr[0,:,:,0,-1]),
                                'AP75':avg(pr[5,:,:,0,-1]),'APs':avg(pr[:,:,:,1,-1]),
                                'APm':avg(pr[:,:,:,2,-1]),'APl':avg(pr[:,:,:,3,-1])},
        'warning':'maxDets=300 results are supplementary and must not replace standard COCO maxDets=100 AP'})
    # Compare formal GT coordinates with the current native validation labels.
    labels=Path('/home/zjnu/文档/ZHJ_WorkSpace/datasets/VisDrone2019-DET-YOLO/VisDrone2019-DET-val/labels')
    anns=defaultdict(list)
    for row in c.dataset['annotations']:anns[row['image_id']].append(row)
    discrepancies=[];maximum=0.;count=0
    for im in c.dataset['images']:
        native=np.array([[float(x) for x in line.split()] for line in (labels/(Path(im['file_name']).stem+'.txt')).read_text().splitlines() if line.strip()]).reshape(-1,5)
        formal=[]
        for row in anns[im['id']]:
            x,y,w,h=row['bbox'];formal.append([row['category_id']-1,(x+w/2)/im['width'],(y+h/2)/im['height'],w/im['width'],h/im['height']])
        formal=np.array(formal).reshape(-1,5);count+=len(native)
        if native.shape!=formal.shape:discrepancies.append({'image_id':im['id'],'label_count':len(native),'formal_count':len(formal)});continue
        native=native[np.lexsort(native.T[::-1])];formal=formal[np.lexsort(formal.T[::-1])]
        maximum=max(maximum,float(np.abs(native-formal).max()) if len(native) else 0.)
    write_json(tracked/'gt_label_audit.json',{'images':len(c.dataset['images']),'native_label_count':count,
        'formal_annotation_count':len(c.dataset['annotations']),'count_discrepancies':discrepancies,
        'max_sorted_normalized_field_difference':maximum,
        'sorting':'lexicographic class,cx,cy,w,h; this is a coordinate sanity check, not formal annotation-ID pairing'})
    image_rows=[];gt_by_image=defaultdict(list)
    for gid,state in result['gt_states'].items():gt_by_image[state['image_id']].append(state)
    for image in sorted(c.dataset['images'],key=lambda x:x['id']):
        states=gt_by_image[image['id']]
        image_rows.append({'image_id':image['id'],'file_name':image['file_name'],'width':image['width'],'height':image['height'],
                           'evaluated_gt':len(states),'matched_iou50':sum(s['matched'][0] for s in states),
                           'matched_iou75':sum(s['matched'][5] for s in states),
                           'density_group':'sparse' if len(states)<=10 else 'moderate' if len(states)<=50 else 'crowded'})
    table(tracked/'per_image_coco.csv',image_rows)
    # Small diagnostic case book with complete selected GT trajectories.
    cases=[];event_counts=Counter();stage_totals=defaultdict(lambda:[0,0.,0.])
    summary=read_json(raw/'baseline/analysis/summary.json')
    for item in summary['images']:
        sample=read_json(raw/'baseline/analysis'/item['sample'])
        for traj in sample['trajectories']['gt']:
            for point in traj['points']:
                if point['valid']:
                    v=stage_totals[point['layer_id']];v[0]+=1;v[1]+=point['iou'];v[2]+=point['gt_class_score']
        for event in sample['events']:
            if event['family']!='gt_object':continue
            for key in ('localization_drop','classification_drop','query_switch'):
                if event[key]:event_counts[key]+=1
        if len(cases)<12:
            for event in sample['events']:
                if event['family']=='gt_object' and event['transition']=='D2->D3' and (event['localization_drop'] or event['classification_drop']):
                    trajectory=next(x for x in sample['trajectories']['gt'] if x['gt_id']==event['gt_id'])
                    cases.append({'event':event,'trajectory':trajectory});break
    write_json(tracked/'diagnostic_cases.json',{'selection':'First 12 image-id ordered GT-object D2->D3 degradations; illustrative, not random sample','cases':cases})
    table(tracked/'gt_stage_means.csv',[{'layer':k,'valid_gt_assignments':v[0],'mean_iou':v[1]/v[0],
                                        'mean_gt_class_probability':v[2]/v[0]} for k,v in sorted(stage_totals.items())])
    bbox_rows=[]
    for payload in iter_batches(raw/'baseline'):
        for image in payload['images']:
            for stage in image['stages']:
                if stage['source']!='decoder':continue
                delta=stage['bbox_delta']
                bbox_rows.append({'image_id':image['metadata']['image_id'],'layer_id':stage['layer_id'],
                    'query_constant_exact':torch.equal(delta,delta[:1].expand_as(delta)),
                    'query_centered_max_abs':float((delta-delta.mean(0,keepdim=True)).abs().max()),
                    'bbox_delta_abs_mean':float(delta.abs().mean()),
                    'normalized_box_step_abs_mean':float((stage['reference_out']-stage['reference_in']).abs().mean())})
    table(tracked/'bbox_output_variation.csv',bbox_rows)
    # Last bbox MLP feature is pre-ReLU; positive fraction equals ReLU activity.
    activity=[];residuals=[]
    for payload in iter_batches(raw/'probe_sample'):
        for image in payload['images']:
            f=image['features'];base=f['model.19'][0];refined=f['model.23'][0]
            residuals.append({'image_id':image['metadata']['image_id'],
                'na_residual_rms_ratio':float((refined-base).norm()/base.norm()),
                'p4_rms':float(base.square().mean().sqrt()),'residual_rms':float((refined-base).square().mean().sqrt())})
            for probe in image.get('probes',[]):
                if probe['kind']=='features' and 'dec_bbox_head.2.layers.1' in str(probe.get('metadata',{})):
                    t=probe['tensors']['values']
                    activity.append({'image_id':image['metadata']['image_id'],'relu_positive_fraction':float((t>0).float().mean()),
                                     'measurement':'pre-ReLU second hidden-layer output; fraction >0'})
    if activity:table(tracked/'bbox_hidden_activity_sample.csv',activity)
    table(tracked/'na_residual_sample.csv',residuals)
    # Freeze a fresh R0 progress snapshot, never copy its moving checkpoint.
    r0=repo/'runs/detect/r18_na_ablation/rtdetr-r18-r0-150e-s0'
    with (r0/'results.csv').open() as f:last=list(csv.DictReader(f))[-1]
    write_json(tracked/'r0_progress_snapshot.json',{'observed_utc':datetime.now(timezone.utc).isoformat(),'last_logged_metrics':last,
        'training_not_complete':int(float(last['epoch']))<150,'not_used_as_completed_comparator':True,
        'initialization':read_json(r0/'initialization.json')})
    artifacts={x.name:{'bytes':x.stat().st_size,'sha256':sha256_file(x)} for x in sorted(tracked.iterdir()) if x.is_file() and x.name not in ('artifact_index.json','README.md','REPORT.md','CHATGPT_CONTEXT.md')}
    write_json(tracked/'artifact_index.json',{'files':artifacts,'scope':'Lightweight source/configuration/summary/case artifacts. Raw tensors, full trajectories, predictions, checkpoints and logs remain ignored under runs/ and logs/.'})
    print('EXPORTED',result['metrics'],'files',len(artifacts),flush=True)

if __name__=='__main__':main()
