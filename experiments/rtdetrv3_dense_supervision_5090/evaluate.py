"""Independent CPU COCOeval; never imports torch."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();out=a.output
    m=json.loads((out/'prediction_metadata.json').read_text());pred=out/'predictions.json'
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(pred)==m['prediction_sha256'] and sha(m['gt_path'])==m['gt_sha256']
    g=COCO(m['gt_path']);ds=json.loads(pred.read_text());ids=sorted({x['image_id'] for x in ds})
    assert len(ids)==m['images'] and len(ds)==len(ids)*300
    result=dict(images=len(ids),detections=len(ds),unit='percent',checkpoint_sha256=m['checkpoint_sha256'],checkpoint_epoch=m['checkpoint_epoch'],protocol=m['protocol'])
    for maximum in [100,300]:
        ev=COCOeval(g,g.loadRes(str(pred)),'bbox');ev.params.imgIds=ids;ev.params.maxDets=[1,10,maximum]
        ev.evaluate();ev.accumulate()
        precision=ev.eval['precision'];recall=ev.eval['recall']
        def mean(v):
            v=v[v>-1];return float(v.mean()*100) if v.size else None
        def ap(area=0,iou=None,cat=None):
            v=precision[:,:,:,area,-1]
            if cat is not None:v=v[:,:,cat]
            if iou is not None:v=v[np.isclose(ev.params.iouThrs,iou)]
            return mean(v)
        metrics=dict(AP=ap(),AP50=ap(iou=.5),AP75=ap(iou=.75),APs=ap(1),APm=ap(2),APl=ap(3),**{f'AR{maximum}':mean(recall[:,:,0,-1])},ARs=mean(recall[:,:,1,-1]),ARm=mean(recall[:,:,2,-1]),ARl=mean(recall[:,:,3,-1]))
        result[f'coco_maxdet{maximum}_percent']=metrics
        if maximum==100:
            result['per_class_maxdet100']=[dict(category_id=int(c),name=g.cats[c]['name'],AP=ap(cat=i),AP50=ap(iou=.5,cat=i),AP75=ap(iou=.75,cat=i),APs=ap(1,cat=i),APm=ap(2,cat=i),APl=ap(3,cat=i),AR100=mean(recall[:,i,0,-1])) for i,c in enumerate(ev.params.catIds)]
        print(json.dumps(dict(maxDet=maximum,**metrics)),flush=True)
    (out/'coco_metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
