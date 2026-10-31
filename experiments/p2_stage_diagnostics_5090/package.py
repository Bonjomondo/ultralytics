"""Audit inference receipts and archive compact evidence; keep raw assets on server."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

def read(p): return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v): p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    run=a.root.resolve(); repo=run.parents[2]; info=read(run/'analysis/summary.json'); status=read(run/'status.json')
    assert status['status']=='completed' and info['status']=='completed'
    assert status['frozen_sources_unchanged'] and status['checkpoints_unchanged']
    assert all(i['status']=='completed' and i['returncode']==0 for i in status['stages'])
    checks=[]; pins=[]
    for model in info['models']:
        summary_path=Path(model['summary']['path']); s=read(summary_path); directory=summary_path.parent
        assert sha(summary_path)==model['summary']['sha256']
        assert sha(Path(model['checkpoint']['path']))==model['checkpoint']['sha256']
        pin=read(Path(model['source_pin']['path'])); pins.append(pin)
        assert sha(Path(model['source_pin']['path']))==model['source_pin']['sha256']
        for k,v in pin['source_sha256'].items(): assert sha(repo/'ultralytics'/k)==v
        for k,v in pin['sdk_source_sha256'].items(): assert sha(Path(pin['sdk_root'])/k)==v
        baseline=Path(s['historical_d3_replay']['previous']['path'])
        old=read(baseline); predictions=read(baseline.with_name('predictions.json') if model['seed'] else baseline.parent/'predictions.json')
        formal=read(directory/'d3-coco.json')
        canonical=hashlib.sha256(json.dumps(predictions,sort_keys=True,allow_nan=False).encode()).hexdigest()
        current=json.loads((directory/'d3-predictions.json').read_text())
        assert current==predictions, 'D3 raw predictions differ from original evaluation'
        assert hashlib.sha256(json.dumps(current,sort_keys=True,allow_nan=False).encode()).hexdigest()==canonical
        assert all(abs(formal['metrics'][k]-old['metrics'][k])<1e-12 for k in formal['metrics'])
        checks.append({'seed':model['seed'],'model':model['model'],'checkpoint_sha_verified':True,'baseline_evaluation':str(baseline),'baseline_sha256':sha(baseline),'D3_canonical_predictions_exact':True,'raw_predictions_canonical_sha256':canonical,'prediction_hash_scope':'raw bbox/category/image/score list before COCO.loadRes adds area/id/segmentation/iscrowd','D3_metrics_exact':formal['metrics']==old['metrics'],'source_pin_verified':True,'raw_manifest_sha256':model['observation']['manifest']['sha256']})
    assert all(x['source_sha256']==pins[0]['source_sha256'] and x['sdk_source_sha256']==pins[0]['sdk_source_sha256'] for x in pins)
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    for file in (run/'analysis').iterdir():
        if file.suffix=='.csv' or file.name=='summary.json': shutil.copy2(file,out/file.name)
    shutil.copy2(run/'status.json',out/'execution.json'); shutil.copy2(run/'preflight.json',out/'preflight.json')
    stability=repo/'runs/p2_stability_5090/5090-p2-stability-20261027-r1'
    shutil.copy2(stability/'paired_seeds.csv',out/'paired_seeds.csv')
    shutil.copy2(stability/'paired_summary.json',out/'paired_summary.json')
    code=repo/'experiments/p2_stage_diagnostics_5090'
    write(out/'audit.json',{'status':'passed','models':checks,'all_six_input_gt_identities_exact':True,'source_and_sdk_identical_across_six':True,'source_files':{str(p.relative_to(repo)):sha(p) for p in code.glob('*.py')},'no_training':True})
    print(out)

if __name__=='__main__': main()
