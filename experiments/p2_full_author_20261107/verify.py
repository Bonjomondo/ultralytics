"""Exercise real-data train, query-tagged prediction, CPU COCOeval and raw recovery."""
import json
import os
from pathlib import Path
import subprocess
import sys

from common import ARMS, EXP, REPORT, RUNS, now, run_name, sha, write
from report import archive, manifest


def stage(label, args, cpu=False):
    path=RUNS/'verification_logs'/f'{label}.log'
    path.parent.mkdir(exist_ok=True)
    env=os.environ.copy()
    if cpu:env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with path.open('w') as stream:
        subprocess.run([sys.executable,'-u',*args],cwd=EXP,env=env,stdout=stream,
                       stderr=subprocess.STDOUT,check=True)


def main():
    for arm in ARMS:
        name=run_name(arm,smoke=True)
        if not (RUNS/name/'weights/last.pt').exists():
            stage(f'train_{arm}',[str(EXP/'train.py'),'--arm',arm,'--smoke'])
        for phase in ['predict','coco']:
            stage(f'{phase}_{arm}',[str(EXP/'evaluate.py'),phase,'--run',name,'--smoke'],phase=='coco')
        archive(name)
    if not (REPORT/'resume_verification.json').exists():
        stage('raw_resume',[str(EXP/'resume_probe.py')])
    stage('benchmark',[str(EXP/'evaluate.py'),'benchmark','--models',
          *[run_name(arm,smoke=True) for arm in ARMS],'--output','smoke_benchmark.json','--smoke'])
    first=[json.loads((RUNS/run_name(arm,smoke=True)/'first_batch.json').read_text()) for arm in ARMS]
    for key in ['images','image_sha256','targets_sha256']:
        assert first[0][key]==first[1][key]==first[2][key],key
    verified={}
    for arm in ARMS:
        metrics=json.loads((RUNS/run_name(arm,smoke=True)/'evaluation/e350_best/metrics.json').read_text())
        assert metrics['images']==16 and metrics['detections']==4800
        assert all(record['gt']>=0 for record in metrics['tiny_recall_maxdet100'])
        verified[arm]={'checkpoint_sha256':metrics['checkpoint_sha256'],
                       'smoke_native_metrics':metrics['native_metrics'],'not_accuracy_evidence':True}
    assert json.loads((REPORT/'resume_verification.json').read_text())['status']=='passed'
    sources={str(path.relative_to(EXP)):sha(path) for path in sorted(EXP.rglob('*'))
             if path.is_file() and path.suffix in ['.py','.yaml'] and '__pycache__' not in path.parts}
    write(REPORT/'smoke_verification.json',{'status':'passed','finished_utc':now(),
          'validated_source_hashes':sources,'first_batch_images_targets_exact':True,
          'losses_can_differ_between_architectures':True,'real_data_smoke':verified,
          'cpu_cocoeval_with_query_origins_and_tiny_recall':True,
          'benchmark':json.loads((RUNS/'smoke_benchmark.json').read_text()),
          'limits':'Short runs verify execution only. They are not precision or convergence evidence.'})
    manifest()
    print('VERIFICATION PASSED',flush=True)


if __name__=='__main__':
    main()
