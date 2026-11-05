"""Seed0 stage-A screening queue; fail closed and archive each formal result."""
from common import *
import subprocess,shutil,traceback,csv
from datetime import datetime,timezone

def now():return datetime.now(timezone.utc).isoformat()

def archive(arm,output):
    dest=REPORT/arm;dest.mkdir(parents=True,exist_ok=True)
    for name in ['coco_metrics.json','prediction_metadata.json']:
        shutil.copy2(output/name,dest/name)
    training=RUNS/'training'/arm
    if training.exists():
        for name in ['receipt.json','initialization.json','source_manifest.json','args.yaml','results.csv','health.jsonl','aux_diagnostics.jsonl']:
            if (training/name).exists():shutil.copy2(training/name,dest/name)
    else:
        shutil.copy2(ROOT/'reference/c2_checkpoint_origin.json',dest/'checkpoint_origin.json')
    (dest/'README.md').write_text(f'# {arm}: author recipe seed0 screening\n\n[Formal COCO metrics](coco_metrics.json) · [Evaluation protocol, checkpoint SHA and forward timing](prediction_metadata.json). Full checkpoints, predictions and logs remain on server at `{output}`.\n')
    manifest(dest)

def manifest(dest):
    (dest/'MANIFEST.sha256').write_text('\n'.join(sha(p)+'  '+str(p.relative_to(dest)) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='MANIFEST.sha256')+'\n')

def summarize():
    arms=['a0_c0','a1_c2_reused','a2_c0_o2mc','a3_c2_o2mc']
    results={k:json.loads((RUNS/'evaluations'/k/'coco_metrics.json').read_text()) for k in arms}
    assert all(v['protocol']==results[arms[0]]['protocol'] for v in results.values())
    rows=[dict(arm=k,**v['coco_maxdet100_percent']) for k,v in results.items()]
    with (REPORT/'stageA_150_seed0.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    m={k:v['coco_maxdet100_percent'] for k,v in results.items()}
    deltas={}
    for base,aux in [('a0_c0','a2_c0_o2mc'),('a1_c2_reused','a3_c2_o2mc')]:
        deltas[aux]={key:m[aux][key]-m[base][key] for key in m[base]}
    interaction={key:deltas['a3_c2_o2mc'][key]-deltas['a2_c0_o2mc'][key] for key in m['a0_c0']}
    eligible=[k for k,v in deltas.items() if v['AP']>0 and v['APs']>=0 and v['AR100']>=0]
    write(REPORT/'screening_decision.json',dict(seed=0,deltas_pp=deltas,interaction_pp=interaction,eligible_for_350_review=eligible,criterion='Single-seed screening: AP>baseline, APs and AR100 do not fall; positive screen is provisional, not stable efficacy.',automatic_boundary='Stage A 150 epoch results and archive only; 350 continuation and B/C implementations require result-based selection and recovery/source gates.',multiseed='deferred by user'))
    text='# 作者配方阶段A：seed0 / 150轮\n\n| 实验 | AP | APs | AR100 |\n|---|---:|---:|---:|\n'
    text+=''.join(f"| {row['arm']} | {row['AP']:.6f} | {row['APs']:.6f} | {row['AR100']:.6f} |\n" for row in rows)
    text+=f"\nO2M-C增益：C0 {deltas['a2_c0_o2mc']['AP']:+.6f}pp，C2 {deltas['a3_c2_o2mc']['AP']:+.6f}pp；交互项 {interaction['AP']:+.6f}pp。单seed不能证明稳定协同。C2复用4090前150轮最佳（epoch145），在5090同环境正式复评；其余三组在5090训练。\n\n[完整精度表](stageA_150_seed0.csv) · [筛选条件及下一阶段候选](screening_decision.json)。350轮是否继续由实际结果决定；旧150轮alpha0.5结果不参与此表。\n"
    (REPORT/'RESULTS.md').write_text(text)
    manifest(REPORT)

def main():
    verify_pin();environment()
    status=dict(status='running',pid=os.getpid(),started_utc=now(),seeds=[0],deferred_seeds=[1,2],stages=[])
    def save():
        write(RUNS/'queue_status.json',status);write(REPORT/'queue_status.json',status)
    def stage(name,args):
        verify_pin();log=RUNS/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
        row=dict(name=name,command=[PYTHON,'-u',*map(str,args)],log=str(log),started_utc=now(),status='running')
        status['active_stage']=name;status['stages'].append(row);save()
        with log.open('a') as f:
            child=subprocess.Popen(row['command'],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy());row['pid']=child.pid;save();rc=child.wait()
        row.update(status='completed' if rc==0 else 'failed',exitcode=rc,finished_utc=now());save()
        assert rc==0,f'{name} failed ({rc}); see {log}'
    def evaluate(name,checkpoint,smoke=False):
        out=RUNS/('smoke-evaluations' if smoke else 'evaluations')/name
        flags=['--smoke'] if smoke else []
        stage(name+'-export'+('-smoke' if smoke else ''),[ROOT/'export.py','--checkpoint',checkpoint,'--output',out,*flags])
        stage(name+'-coco'+('-smoke' if smoke else ''),[ROOT/'evaluate.py','--output',out])
        if not smoke:archive(name,out)
    save()
    try:
        stage('engineering-gates',[ROOT/'gates.py'])
        for arm in ['a2_c0_o2mc','a3_c2_o2mc']:
            stage(arm+'-smoke-train',[ROOT/'train.py','--arm',arm,'--smoke'])
            evaluate(arm,RUNS/'smoke'/arm/'weights/best.pt',True)
        evaluate('a1_c2_reused',REPO.parent/'reproductions/rtdetrv3_dense_supervision_5090/5090-ds01-c2-author-epoch150-best-native.pt')
        for arm in ['a0_c0','a2_c0_o2mc','a3_c2_o2mc']:
            stage(arm+'-train150',[ROOT/'train.py','--arm',arm])
            evaluate(arm,RUNS/'training'/arm/'weights/best.pt')
        summarize();status.update(status='stageA_150_complete_review_for_350',active_stage=None,finished_utc=now());save();manifest(REPORT)
    except BaseException:
        status.update(status='failed',error=traceback.format_exc(),finished_utc=now());save();raise
if __name__=='__main__':main()
