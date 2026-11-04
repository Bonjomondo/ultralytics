"""Locked author-recipe scalar experiment: train, predict, CPU evaluate, compare."""
from pathlib import Path
import csv,fcntl,json,os,subprocess,sys,hashlib,shutil,traceback
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/c2-scalar-author-350e-s0'
REPORT=Path('/home/zjnu/文档/ZHJ_WorkSpace/ultralytics/docs/experiment_reports/2026-11-04_scalar_author350')

def now():return datetime.now(timezone.utc).isoformat()
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def finish():
    m=json.loads((RUN/'evaluation/coco_metrics.json').read_text());bench=json.loads((RUN/'evaluation/benchmark.json').read_text())
    b=ROOT.parent;references={'B0':b/'rtdetr-r18-sodetr-paper-20261027/runs/rtdetr-r18-author-recipe-350e-s0/evaluation/metrics.json','C2':b/'rtdetr-r18-c2-author-recipe-20261028/runs/rtdetr-r18-c2-matched-b0-350e-s0/evaluation/coco_metrics.json'}
    records={k:json.loads(p.read_text()) for k,p in references.items()};records['scalar']=m
    assert all(d['gt_sha256']==m['gt_sha256'] for d in records.values())
    assert sha(RUN/'weights/best.pt')==m['checkpoint_sha256']
    rows=[]
    for label,d in records.items():
        metrics=d.get('coco_maxdet100_percent',d.get('coco_maxdet100'))
        assert d['checkpoint_sha256']==bench['models'][label]['checkpoint_sha256']
        rows.append({'model':label,**{k:metrics[k] for k in ['AP','AP50','AP75','APs','APm','APl']},'forward_median_ms':bench['models'][label]['median_ms']})
    effects={label:{k:rows[-1][k]-row[k] for k in ['AP','AP50','AP75','APs','APm','APl']} for label,row in zip(['scalar-B0','scalar-C2'],rows[:2])}
    write(ROOT/'final_result.json',{'finished_utc':now(),'rows':rows,'effects_pp':effects,'checkpoint':m,'benchmark':bench,'limitations':['single seed','historical B0/B1 full initial states not archived; reconstructed common initialization verified','350 epoch budget/patience40; no guarantee of convergence','FP32 atomic CPU checkpoint serialization differs; train math and workers8 retained']})
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    for name in ['results.csv','initialization.json','status.json','source_manifest.json']:
        shutil.copyfile(RUN/name,REPORT/name)
    for p in [ROOT/'final_result.json',RUN/'evaluation/coco_metrics.json',RUN/'evaluation/benchmark.json']:
        shutil.copyfile(p,REPORT/p.name)
    status=json.loads((RUN/'status.json').read_text());status.update(status='complete',formal_metrics=m['coco_maxdet100_percent'],best_epoch=m['checkpoint_epoch']);write(RUN/'status.json',status);write(REPORT/'status.json',status)
    lines=['# Scalar author-engine 350轮对照结果','','|模型|AP|AP50|APs|GPU前向中位ms|','|---|---:|---:|---:|---:|']
    lines += [f"|{r['model']}|{r['AP']:.3f}|{r['AP50']:.3f}|{r['APs']:.3f}|{r['forward_median_ms']:.3f}|" for r in rows]
    lines += ['','差值（百分点）：'+json.dumps(effects,ensure_ascii=False),'','单seed；历史初始化重放限制见README。COCO maxDets100，converted VisDrone val548；速度不含预后处理。']
    (REPORT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
def main():
    with (ROOT/'controller.lock').open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (RUN/'weights/last.pt').exists(),'Existing run: explicit recovery required'
        manifest=json.loads((ROOT/'source_manifest.json').read_text())
        status={'status':'running','started_utc':now(),'pid':os.getpid(),'stages':[]}
        try:
            for phase,cmd in [('train',['train_scalar.py']),('predict',['export_predictions.py','--run',str(RUN)]),('coco',['coco_eval.py','--run',str(RUN)]),('benchmark',['benchmark.py','--run',str(RUN)])]:
                assert all(sha(ROOT/k)==v for k,v in manifest.items()),'Frozen sources changed'
                item={'phase':phase,'status':'running','started_utc':now()};status['stages'].append(item);write(ROOT/'controller_status.json',status)
                env=os.environ.copy()
                if phase=='coco':env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                with (ROOT/(phase+'.log')).open('w') as log:
                    process=subprocess.Popen([sys.executable,'-u',*cmd],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);status['child_pid']=process.pid;write(ROOT/'controller_status.json',status);code=process.wait()
                item.update(status='complete' if code==0 else 'failed',returncode=code,finished_utc=now());write(ROOT/'controller_status.json',status)
                if code:raise RuntimeError(f'{phase} failed ({code}); see {phase}.log')
            finish();status.update(status='complete',finished_utc=now(),child_pid=None);write(ROOT/'controller_status.json',status)
        except BaseException:
            status.update(status='failed',error=traceback.format_exc());write(ROOT/'controller_status.json',status);raise
if __name__=='__main__':main()
