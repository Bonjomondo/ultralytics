"""Sequential seed1/seed2 training, formal paired evaluation, and auto archival."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from experiments.c0_o2mc_5090 import run as original
from experiments.c0_o2mc_5090.support import check_reference, summarize, manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root,output=args.root.resolve(),args.output.resolve()
    if (root/'queue_status.json').exists():raise FileExistsError('Never overwrite/resume an existing queue')
    root.mkdir(parents=True,exist_ok=True)
    env=original.configure()
    reference=check_reference()
    pin=original.source_hashes()
    original.write_atomic(root/'queue_plan.json',{'environment':env,'reference_seed0':reference,'source_sha256':pin,
        'seeds':[0,1,2],'root':str(root),'archive':str(output),'cwd':str(original.ROOT),
        'tmux':os.environ.get('TMUX'),'created_utc':original.now(),'launch':str(root/'launch.sh')})
    state={'status':'running','pid':os.getpid(),'started_utc':original.now(),'seeds':[0,1,2],
           'completed_seeds':[],'active_seed':None,'children':[]}
    def save():
        original.write_atomic(root/'queue_status.json',state)
        (output/'runtime').mkdir(exist_ok=True)
        shutil.copy2(root/'queue_status.json',output/'runtime/queue_status.json')
        for name in ['queue_plan.json','launch.sh']:
            if (root/name).exists():shutil.copy2(root/name,output/'runtime'/name)
    save()
    try:
        for seed in (0,1,2):
            if original.source_hashes()!=pin:raise RuntimeError('Queue frozen source changed')
            run=root/f'seed{seed}'
            run.mkdir(exist_ok=False)
            command=[sys.executable,'-u','-m','experiments.c0_o2mc_5090.run','pipeline',
                     '--seed',str(seed),'--root',str(run),'--output',str(output/f'seed{seed}')]
            launch='#!/usr/bin/env bash\nset -euo pipefail\n'
            import shlex
            launch+='cd '+shlex.quote(str(original.ROOT))+'\nexport PYTHONPATH="$PWD:$PWD/../detrscope/src"\n'
            launch+='exec '+shlex.join(command)+'\n'
            (run/'launch.sh').write_text(launch)
            state.update(active_seed=seed)
            log=run/'controller.log'
            with log.open('x') as stream:
                proc=subprocess.Popen(['bash',str(run/'launch.sh')],cwd=original.ROOT,stdout=stream,stderr=subprocess.STDOUT)
                record={'seed':seed,'pid':proc.pid,'command':command,'log':str(log),'started_utc':original.now()}
                state['children'].append(record);save()
                code=proc.wait()
            record.update(returncode=code,finished_utc=original.now())
            if code:raise RuntimeError(f'seed{seed} pipeline failed: {log}')
            status=json.loads((run/'status.json').read_text())
            if status['status']!='completed':raise RuntimeError('Missing pipeline completion receipt')
            arc=output/f'seed{seed}'
            for name in ['status.json','plan.json','launch.sh','exit.json']:
                if (run/name).exists():shutil.copy2(run/name,arc/name)
            gates=arc/'gates';gates.mkdir(exist_ok=True)
            for name in ['dataset-audit','cpu-gates','smoke-eval','full-eval','baseline-eval']:
                shutil.copy2(run/f'logs/{name}.log',gates/f'{name}.log')
            shutil.copy2(run/'training'/f'5090-c0-o2mc-smoke-s{seed}'/'experiment.json',gates/'smoke_training.json')
            shutil.copy2(run/'smoke-evaluation/summary.json',gates/'smoke_evaluation.json')
            manifest(arc)
            state['completed_seeds'].append(seed);save();summarize(output)
        state.update(status='completed',active_seed=None,finished_utc=original.now());save();manifest(output)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc(),finished_utc=original.now());save();manifest(output)
        raise


if __name__=='__main__':main()
