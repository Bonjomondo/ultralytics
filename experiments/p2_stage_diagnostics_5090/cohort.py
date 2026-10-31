"""Track the same GT's cross-model coverage advantage through four stages."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from pycocotools.coco import COCO
from experiments.p2_stability_5090.same_gt import attributes

def main():
    p=argparse.ArgumentParser(); p.add_argument('--analysis',type=Path,required=True); a=p.parse_args()
    repo=Path(__file__).resolve().parents[2]
    gt=COCO(str(repo.parent/'SO_DETR/datasets/VisDrone2019-DET-YOLO/annotations/VisDrone2019-DET_val_coco.json'))
    attrs=attributes(gt); rows=[]
    for seed in (0,1,2):
        models={arm:json.loads((a.analysis/f's{seed}-{arm}-coverage-gt.json').read_text()) for arm in ('c0','c2')}
        assert models['c0'].keys()==models['c2'].keys()
        for iou in (.5,.75):
            for kind in ('correct',):
                states={}
                for stage in ('E','D1','D2','D3'):
                    states[stage]={}
                    for gid in models['c0']:
                        x,y=[models[arm][gid][f'{stage}:{iou:g}'][kind] for arm in ('c0','c2')]
                        states[stage][gid]='both' if x and y else 'c0_only' if x else 'c2_only' if y else 'neither'
                for before,after in (('E','D1'),('D1','D2'),('D2','D3'),('E','D3')):
                    count=Counter()
                    for gid in states[before]:
                        for dim in ('all','class','coco_size'):
                            count[dim,attrs[int(gid)][dim],states[before][gid],states[after][gid]]+=1
                    for (dim,group,origin,dest),number in sorted(count.items()):
                        denom=sum(v for (d,g,o,t),v in count.items() if (d,g,o)==(dim,group,origin))
                        rows.append({'seed':seed,'iou':iou,'kind':kind,'transition':before+'->'+after,'dimension':dim,'group':group,'from_advantage_state':origin,'to_advantage_state':dest,'gt_count':number,'from_cohort_denominator':denom,'fraction':number/denom})
    with (a.analysis/'advantage_cohorts.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('Cohort rows',len(rows))

if __name__=='__main__': main()
