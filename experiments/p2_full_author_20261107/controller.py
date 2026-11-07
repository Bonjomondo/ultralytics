"""Bounded detached queue for three fresh 350-epoch P2 experiments."""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

from common import ARMS, BRANCH, EXP, MILESTONES, REPO, REPORT, RUNS, dataset_identity, now, run_name, sha, write
from report import analyze, archive, manifest, final_audit


def publish(message):
    manifest()
    if subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()!=BRANCH:
        write(RUNS/'publication_pending.json',{'reason':'repository branch changed','updated_utc':now()})
        return
    if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip():
        write(RUNS/'publication_pending.json',{'reason':'other staged work','updated_utc':now()})
        return
    subprocess.run(['git','add',str(REPORT.relative_to(REPO))],cwd=REPO,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO).returncode:
        subprocess.run(['git','commit','-m',message],cwd=REPO,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    write(RUNS/'publication_pending.json',{'branch':BRANCH,'commit':commit,'updated_utc':now()})


def main():
    with (RUNS/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=json.loads((RUNS/'source_identity.json').read_text())
        assert os.environ['AUTHOR_P2_MANIFEST_SHA']==sha(RUNS/'source_identity.json')
        path=RUNS/'queue_status.json'
        state=json.loads(path.read_text()) if path.exists() else {'started_utc':now(),'events':[],'done':[]}
        if state.get('status')=='complete':
            return
        state.update(status='running',pid=os.getpid(),child_pid=None,updated_utc=now())
        state.pop('error',None)
        write(path,state)

        def verify():
            assert all(sha(EXP/name)==expected for name,expected in identity['manifest'].items()),'frozen source changed'

        def event(label, arguments, train=False):
            if label in state['done']:
                return
            run=RUNS/label.split('/')[0]
            if train and (run/'status.json').exists():
                status=json.loads((run/'status.json').read_text())
                if status.get('status') in ['training_complete_pending_evaluation','complete']:
                    state['done'].append(label);write(path,state);return
            budget=2 if train else 3
            previous=sum(item['label']==label for item in state['events'])
            for attempt in range(previous+1,budget+1):
                verify()
                command=list(arguments)
                if train:
                    assert dataset_identity()==json.loads((REPORT/'dataset_identity.json').read_text())
                    last=run/'weights/last.pt'
                    if last.exists():
                        saved=RUNS/'recovery_snapshots'/run.name/str(time.time_ns())
                        saved.mkdir(parents=True)
                        for source in [last,run/'weights/best.pt',run/'status.json',run/'failure.json']:
                            if source.exists():shutil.copyfile(source,saved/source.name)
                        command.append('--resume')
                    elif (run/'results.csv').exists():
                        raise RuntimeError('Curve exists without durable checkpoint')
                item={'label':label,'attempt':attempt,'started_utc':now(),'status':'running'}
                state['events'].append(item)
                log=RUNS/'logs'/f"{label.replace('/','-')}-attempt{attempt}.log"
                log.parent.mkdir(exist_ok=True)
                item['log']=str(log)
                env=os.environ.copy()
                if '/coco/' in label:
                    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                with log.open('w') as output:
                    child=subprocess.Popen([sys.executable,'-u',*command],cwd=EXP,env=env,
                                           stdout=output,stderr=subprocess.STDOUT)
                    state.update(current=label,child_pid=child.pid,updated_utc=now());write(path,state)
                    snapshot_epoch=-1
                    while child.poll() is None:
                        time.sleep(15)
                        if train:
                            status_path=run/'status.json'
                            age=time.time()-max(log.stat().st_mtime,status_path.stat().st_mtime if status_path.exists() else 0)
                            if age>1800:
                                child.terminate();item['stale_process_terminated']=True
                                try:child.wait(timeout=30)
                                except subprocess.TimeoutExpired:child.kill()
                            if status_path.exists():
                                complete=json.loads(status_path.read_text()).get('completed_epochs',0)
                                if complete and complete%25==0 and complete!=snapshot_epoch:
                                    archive(run.name);analyze();publish(f'Archive running P2 experiment {run.name} epoch{complete}')
                                    snapshot_epoch=complete
                    code=child.wait()
                item.update(status='complete' if code==0 else 'failed',returncode=code,finished_utc=now())
                state.update(child_pid=None,updated_utc=now());write(path,state)
                if code==0:
                    state['done'].append(label);write(path,state);return
                if train and not last.exists():
                    raise RuntimeError(f'{label} failed before checkpoint; see {log}')
            raise RuntimeError(f'{label} exhausted its {budget} total attempts; inspect preserved evidence')

        try:
            for arm in ARMS:
                name=run_name(arm)
                if f'{name}/archived' in state['done']:
                    continue
                event(f'{name}/train',[str(EXP/'train.py'),'--arm',arm],True)
                for epoch in MILESTONES:
                    for kind in ['last','best']:
                        for phase in ['predict','coco']:
                            event(f'{name}/{phase}/e{epoch}_{kind}',[str(EXP/'evaluate.py'),phase,
                                  '--run',name,'--budget',str(epoch),'--kind',kind])
                status=json.loads((RUNS/name/'status.json').read_text())
                status.update(status='complete');write(RUNS/name/'status.json',status)
                archive(name);analyze();publish(f'Archive complete P2 experiment {name}')
                state['done'].append(f'{name}/archived');write(path,state)
            event('final/benchmark',[str(EXP/'evaluate.py'),'benchmark','--models',
                  *[run_name(arm) for arm in ARMS],'--output','benchmark_final.json'])
            shutil.copyfile(RUNS/'benchmark_final.json',REPORT/'benchmark_final.json')
            final_audit()
            state.update(status='complete',finished_utc=now(),child_pid=None)
            write(path,state);write(REPORT/'final_queue_status.json',state)
            analyze();publish('Finalize three-arm full-P2 author experiment')
        except BaseException:
            state.update(status='failed',error=traceback.format_exc(),updated_utc=now())
            write(path,state);write(REPORT/'failure.json',state)
            for arm in ARMS:
                if (RUNS/run_name(arm)).exists():archive(run_name(arm))
            analyze();manifest()
            raise


if __name__=='__main__':
    main()
