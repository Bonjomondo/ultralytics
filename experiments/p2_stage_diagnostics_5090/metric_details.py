"""Offline IoU/class AP decomposition and final-score-budget sensitivity."""
import argparse
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from experiments.p2_stability_5090.same_gt import match,attributes,compare

def table(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def avg(v):
    v=v[v>-1]; return float(v.mean())*100 if v.size else None
def main():
    p=argparse.ArgumentParser(); p.add_argument('--analysis',type=Path,required=True); a=p.parse_args()
    summary=json.loads((a.analysis/'summary.json').read_text()); gt=COCO(summary['annotations']['path'])
    groups=attributes(gt); ious=[]; classes=[]; states={}
    for model in summary['models']:
        directory=Path(model['summary']['path']).parent
        for depth in (1,2,3):
            predictions=json.loads((directory/f'd{depth}-predictions.json').read_text())
            with redirect_stdout(io.StringIO()):
                detections=gt.loadRes(predictions); ev=COCOeval(gt,detections,'bbox')
                ev.params.imgIds=sorted(gt.imgs); ev.params.maxDets=[1,10,100]
                ev.evaluate(); ev.accumulate(); ev.summarize()
            formal=json.loads((directory/f'd{depth}-coco.json').read_text())
            assert abs(float(ev.stats[0])-formal['metrics']['AP'])<1e-12
            base={'seed':model['seed'],'model':model['model'],'stage':f'D{depth}'}
            for ti,threshold in enumerate(ev.params.iouThrs):
                ious.append({**base,'iou':float(threshold),'AP':avg(ev.eval['precision'][ti,:,:,0,-1])})
            for ci,category in enumerate(ev.params.catIds):
                classes.append({**base,'category':gt.cats[category]['name'],**{name:avg(ev.eval['precision'][:,:,ci,area,-1]) for area,name in enumerate(('AP','APs','APm','APl'))}})
            if depth==3:
                with redirect_stdout(io.StringIO()): states[model['seed'],model['model']]=match(gt,predictions)
            print('DETAIL',model['seed'],model['model'],depth,flush=True)
    table(a.analysis/'per_iou_metrics.csv',ious); table(a.analysis/'class_metrics.csv',classes)
    sensitivity=[]
    for seed in (0,1,2):
        for iou in (.5,.75):
            for budget in (100,300):
                for conf in (.001,.05,.25):
                    rows=compare(states[seed,'c0'][iou,budget],states[seed,'c2'][iou,budget],groups,conf)
                    main=next(x for x in rows if x['dimension']=='all')
                    sensitivity.append({'seed':seed,'stage':'D3','iou':iou,'max_dets_per_category':budget,'conf':conf,**main})
    table(a.analysis/'final_budget_sensitivity.csv',sensitivity)

if __name__=='__main__': main()
