"""Pool same-protocol three-seed stage evidence without training or model changes."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

import torch
from pycocotools.coco import COCO
from detrscope.core.records import iter_batches
from experiments.decoder_budget_5090.coverage import coverage_for_image
from experiments.decoder_budget_5090.run import summarize_coverage, input_identity
from experiments.decoder_budget_5090.screen_decoder_eval_5090 import artifact, pooled_formal_changes
from experiments.p2_stability_5090.same_gt import attributes, compare

REPO=Path(__file__).resolve().parents[2]
STAGES=('E','D1','D2','D3')
def read(p): return json.loads(p.read_text())
def write(p,v): p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)

def collect(directory,seed,arm,out):
    summary=read(directory/'summary.json')
    assert summary['status']=='completed' and summary['model_state_unchanged'] and summary['hooks_remaining']==0
    assert summary['early_exit_equivalence']['all_exact'] and summary['observation']['all_identity_and_restore_passed']
    assert summary['protocol']['images']==548 and summary['protocol']['batch']==2
    raw=directory/'observed-d3'
    identity=input_identity(raw)
    assert identity==summary['observation']['input_gt_identity_sha256']
    pools=defaultdict(list); states={}; images=0
    for payload in iter_batches(raw):
        for image in payload['images']:
            images+=1
            targets=image['targets']
            for stage in image['stages']:
                name='E' if stage['source']=='encoder' else f"D{stage['layer_id']}"
                assert name in STAGES
                # All 300 candidates are retained; sorting only satisfies the
                # reusable coverage helper's prefix contract, never changes Q.
                order=stage['logits'].max(-1).values.argsort(descending=True)
                ordered={**stage,'boxes':stage['boxes'][order],'logits':stage['logits'][order]}
                if 'probabilities' in stage: ordered['probabilities']=stage['probabilities'][order]
                temp={**image,'stages':[{**ordered,'source':'encoder'}]}
                c=coverage_for_image(temp,budgets=(300,),thresholds=(.3,.5,.75))
                pools[name].append(c)
                for row in c['per_gt']:
                    if not row['eligible']: continue
                    gid=str(row['gt_id']); key=f"{name}:{row['iou_threshold']:g}"
                    entry=states.setdefault(gid,{'image_id':image['metadata']['image_id'],'class_index':row['class_index']})
                    assert key not in entry
                    entry[key]={'geometry':row['first_covered_rank'] is not None,'correct':row['first_class_correct_rank'] is not None}
        if images%100==0: print('COVERAGE',seed,arm,images,flush=True)
    assert images==548 and len(states)==38759
    coverage=[]
    for stage in STAGES:
        for row in summarize_coverage(pools[stage])['rows']:
            coverage.append({'seed':seed,'model':arm,'stage':stage,**row})
    write(out/f's{seed}-{arm}-coverage-gt.json',states)
    write(out/f's{seed}-{arm}-coverage-summary.json',coverage)
    formal={}
    for d in (1,2,3):
        item=summary['conditions'][f'd{d}']
        path=directory/f'd{d}-coco.json'
        assert artifact(path)==item['coco']
        assert artifact(directory/f'd{d}-predictions.json')==item['predictions']
        formal[f'D{d}']=read(path)
        assert formal[f'D{d}']['metrics']==item['metrics']
    return summary,identity,coverage,states,formal

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    torch.set_num_threads(4)
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    annotations=REPO.parent/'SO_DETR/datasets/VisDrone2019-DET-YOLO/annotations/VisDrone2019-DET_val_coco.json'
    gt=COCO(str(annotations)); groups=attributes(gt); anns={str(k):v for k,v in gt.anns.items()}
    all_cov=[]; metrics=[]; identities=[]; models={}; receipts=[]; diag=[]
    for seed in (0,1,2):
        for arm in ('c0','c2'):
            directory=REPO/f'runs/diagnostics/5090-depth-eval-{arm}-20261025-final' if seed==0 else a.root/f'full-s{seed}-{arm}'
            summary,identity,cov,states,formal=collect(directory,seed,arm,out)
            identities.append(identity); all_cov+=cov; models[seed,arm]=(states,formal)
            receipts.append({'seed':seed,'model':arm,'reused':seed==0,'summary':artifact(directory/'summary.json'),'checkpoint':summary['checkpoint'],'input_gt_identity_sha256':identity,'source_pin':summary['protocol']['source_pin'],'observation':summary['observation'],'early_exit_equivalence':summary['early_exit_equivalence'],'model_state_unchanged':summary['model_state_unchanged']})
            for stage,d in formal.items(): metrics.append({'seed':seed,'model':arm,'stage':stage,**{k:100*v for k,v in d['metrics'].items()}})
            if (directory/'diagnostic_transitions.csv').exists():
                diag += [{'seed':seed,'model':arm,**r} for r in csv.DictReader((directory/'diagnostic_transitions.csv').open())]
    assert len(set(identities))==1, 'Input/GT/ordering differs across models or seeds'
    table(out/'coverage.csv',all_cov); table(out/'formal_metrics.csv',metrics); table(out/'sdk_trajectory_transitions.csv',diag)
    deltas=[]
    for seed in (0,1,2):
        for stage in ('D1','D2','D3'):
            left=next(r for r in metrics if r['seed']==seed and r['stage']==stage and r['model']=='c0')
            right=next(r for r in metrics if r['seed']==seed and r['stage']==stage and r['model']=='c2')
            deltas.append({'seed':seed,'stage':stage,**{f'delta_{k}_pp':right[k]-left[k] for k in left if k not in ('seed','stage','model')}})
    table(out/'paired_stage_deltas.csv',deltas)
    covdiff=[]
    for seed in (0,1,2):
        for left in [r for r in all_cov if r['seed']==seed and r['model']=='c0']:
            right=next(r for r in all_cov if r['seed']==seed and r['model']=='c2' and all(r[k]==left[k] for k in ('stage','group','iou_threshold','budget')))
            assert left['denominator']==right['denominator']
            covdiff.append({k:left[k] for k in ('seed','stage','group','iou_threshold','budget','denominator')}|{'geometry_delta_pp':100*(right['coverage']-left['coverage']) if left['coverage'] is not None else None,'correct_delta_pp':100*(right['class_correct_coverage']-left['class_correct_coverage']) if left['coverage'] is not None else None})
    table(out/'paired_coverage_deltas.csv',covdiff)
    changes=[]; covchanges=[]
    for seed in (0,1,2):
        pairs=[(('c0',s),('c2',s)) for s in STAGES]+[((arm,l),(arm,r)) for arm in ('c0','c2') for l,r in zip(STAGES,STAGES[1:])]
        for (arm1,s1),(arm2,s2) in pairs:
            label=f'{arm1}:{s1}->{arm2}:{s2}'
            if s1!='E' and s2!='E':
                _,pooled=pooled_formal_changes(models[seed,arm1][1][s1],models[seed,arm2][1][s2],anns,label)
                changes += [{'seed':seed,**r} for r in pooled]
            for threshold in (.3,.5,.75):
                for kind in ('geometry','correct'):
                    left={int(g):{'score':1.,'iou':threshold} for g,v in models[seed,arm1][0].items() if v[f'{s1}:{threshold:g}'][kind]}
                    right={int(g):{'score':1.,'iou':threshold} for g,v in models[seed,arm2][0].items() if v[f'{s2}:{threshold:g}'][kind]}
                    covchanges += [{'seed':seed,'transition':label,'iou':threshold,'kind':kind,**r} for r in compare(left,right,groups,0.)]
    table(out/'formal_gt_transitions.csv',changes); table(out/'coverage_gt_transitions.csv',covchanges)
    aggregate=[]
    for stage in ('D1','D2','D3'):
        for metric in ('AP','APs','AP50','AP75','APm','APl'):
            vals=[r[f'delta_{metric}_pp'] for r in deltas if r['stage']==stage]
            aggregate.append({'stage':stage,'metric':metric,'mean_delta_pp':statistics.mean(vals),'sample_std_pp':statistics.stdev(vals),'positive_seeds':sum(x>0 for x in vals),'min_pp':min(vals),'max_pp':max(vals)})
    table(out/'paired_statistics.csv',aggregate)
    write(out/'summary.json',{'status':'completed','images':548,'valid_gt':len(groups),'seeds':[0,1,2],'annotations':artifact(annotations),'input_gt_identity_sha256':identities[0],'models':receipts,'no_training':True,'protocol':{'precision':'native unfused FP32; autocast off','preprocess':'scale_fill640 RGB/255','formal':'COCO bbox maxDets [1,10,100]; per-query argmax; conf .001; max300; no NMS/rounding','coverage':'All300 selected queries; many-to-many, class-agnostic and argmax-class-correct; IoU .3/.5/.75; geometry/coverage not AP','comparison':'same GT id across arms; no cross-model query id association','scope':'observational stage localization; does not isolate P2 causal effect or compare independently trained shallow models'}})
    print('PAIRED_STAGE_DELTAS',json.dumps(deltas),flush=True)
    print('COVERAGE_ALL',json.dumps([r for r in covdiff if r['group']=='all']),flush=True)

if __name__=='__main__': main()
