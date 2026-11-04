"""Audited raw-state recovery after the author scalar DataLoader worker SIGSEGV."""
from pathlib import Path
import argparse,csv,hashlib,json,os,random,shutil,subprocess,sys,time,traceback,fcntl
import numpy as np
import torch
from safe_train import SafeTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import callbacks
from controller import ROOT,RUN,REPORT,write,now,sha,finish

ARCHIVE=ROOT/'worker_failure_130'
CP=ARCHIVE/'last.pt'

def prepare():
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    ARCHIVE.mkdir(exist_ok=False)
    for p in [RUN/'weights/last.pt',RUN/'weights/best.pt',RUN/'results.csv',RUN/'status.json',RUN/'args.yaml',ROOT/'controller_status.json']:
        if p.exists():shutil.copy2(p,ARCHIVE/('controller_status.json' if p==ROOT/'controller_status.json' else p.name))
    c=torch.load(CP,map_location='cpu',weights_only=False)
    assert c['epoch']==129 and c['optimizer'] is not None and c['ema'] is not None and c['rng']
    assert all(torch.isfinite(v).all() for k in ['model','ema'] for v in c[k].state_dict().values())
    assert len(list(csv.DictReader((RUN/'results.csv').open())))==130
    write(ARCHIVE/'receipt.json',{'checkpoint_epoch':130,'last_sha256':sha(CP),'best_sha256':sha(ARCHIVE/'best.pt'),
          'cause':'DataLoader worker killed by SIGSEGV at epoch131; 130 complete epochs preserved',
          'change':'training workers8 -> workers0; raw model, EMA, optimizer, scaler, stopper and RNG restored',
          'limits':'worker concurrency and DataLoader order/augmentation RNG streams differ; not bitwise equivalent continuation',
          'original_sources_unchanged':True,'recovery_script_sha256':sha(Path(__file__))})

class RecoveryTrainer(SafeTrainer):
    def check_resume(self, overrides):
        assert Path(self.args.resume).resolve()==CP.resolve()
        self.resume=True

    def setup_model(self):
        c=torch.load(CP,map_location='cpu',weights_only=False)
        self.model=RTDETRDetectionModel(c['model'].yaml,nc=10,verbose=False)
        self.model.load_state_dict(c['model'].float().state_dict(),strict=True)
        assert all(torch.equal(v,c['model'].state_dict()[k]) for k,v in self.model.state_dict().items())
        return c

    def resume_training(self,c):
        super().resume_training(c)
        self.scheduler.load_state_dict(c['scheduler']);self.scaler.load_state_dict(c['scaler']);self.stopper.__dict__.update(c['stopper'])
        assert self.start_epoch==130
        current=self.optimizer.state_dict()
        for key,state in c['optimizer']['state'].items():
            for name,v in state.items():
                got=current['state'][key][name]
                assert torch.equal(got.cpu(),v.cpu()) if isinstance(v,torch.Tensor) else got==v
        assert self.ema.updates==c['updates']
        assert all(torch.equal(v.cpu(),c['ema'].state_dict()[k].cpu()) for k,v in self.ema.ema.state_dict().items())
        write(ROOT/'recovery_verification.json',{'verified_utc':now(),'resume_epoch_one_based':131,
              'raw_model_exact':True,'optimizer_exact':True,'ema_exact':True,'scaler_restored':True,
              'stopper':vars(self.stopper),'checkpoint_sha256':sha(CP),'workers':self.args.workers,
              'limits':'RNG states restored, but DataLoader worker streams/order differ after workers8->0'})

    def final_eval(self):pass

def train(verify_only=False):
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    assert all(sha(ROOT/k)==v for k,v in manifest.items())
    torch.set_num_threads(8);callbacks.add_integration_callbacks=lambda instance:None
    c=torch.load(CP,map_location='cpu',weights_only=False)
    args=dict(c['train_args']);args.pop('save_dir',None)
    args.update(resume=str(CP),model=str(CP),workers=0,project=str(RUN.parent),name=RUN.name,exist_ok=True)
    assert args['epochs']==350 and args['patience']==40 and args['batch']==4 and not args['amp']
    rng=c['rng'];del c
    t=RecoveryTrainer(overrides=args)
    if verify_only:
        t._setup_train(1)
        print('RAW_RESTORE_VERIFIED',flush=True);return
    def restore_rng(trainer):
        torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])
    t.add_callback('on_train_start',restore_rng)
    batches=0
    def record(trainer,end=False):
        nonlocal batches
        if not end:
            batches+=1
            if batches!=1 and batches%100:return
        write(RUN/'status.json',{'status':'training','pid':os.getpid(),'current_epoch':trainer.epoch+1,
              'completed_epochs':trainer.epoch+1 if end else trainer.epoch,'batches_since_resume':batches,
              'updated_utc':now(),'workers':0,'internal_metrics':trainer.metrics,
              'gate_multiplier':float((2*trainer.model.model[18].logit.sigmoid()).detach()),
              'recovery_checkpoint_sha256':sha(CP),'resume_note':'raw state at epoch130; worker concurrency changed'})
    t.add_callback('on_train_batch_end',record);t.add_callback('on_fit_epoch_end',lambda t:record(t,True))
    t.train()
    write(RUN/'status.json',{'status':'training_complete_pending_evaluation','completed_epochs':t.epoch+1,
          'updated_utc':now(),'workers':0,'recovered_from_epoch':130})

def controller():
    with (ROOT/'recovery.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        s={'status':'running','recovery':True,'started_utc':now(),'stages':[]}
        try:
            for phase,cmd in [('train',[str(Path(__file__)),'--train']),('predict',['export_predictions.py','--run',str(RUN)]),('coco',['coco_eval.py','--run',str(RUN)]),('benchmark',['benchmark.py','--run',str(RUN)])]:
                stage={'phase':phase,'status':'running'};s['stages'].append(stage);write(ROOT/'controller_status.json',s)
                env=os.environ.copy()
                if phase=='coco':env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                with (ROOT/('recovery_'+phase+'.log')).open('w') as f:
                    child=subprocess.Popen([sys.executable,'-u',*cmd],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                    s['child_pid']=child.pid;write(ROOT/'controller_status.json',s);code=child.wait()
                stage.update(status='complete' if code==0 else 'failed',returncode=code,finished_utc=now())
                if code:raise RuntimeError(f'recovery {phase} failed: {code}')
            finish()
            p=ROOT/'final_result.json';d=json.loads(p.read_text());d['recovery_receipt']=json.loads((ARCHIVE/'receipt.json').read_text());d['restore_verification']=json.loads((ROOT/'recovery_verification.json').read_text());write(p,d)
            shutil.copyfile(p,REPORT/'final_result.json')
            shutil.copyfile(ROOT/'recovery_verification.json',REPORT/'recovery_verification.json')
            s.update(status='complete',finished_utc=now(),child_pid=None);write(ROOT/'controller_status.json',s)
        except BaseException:
            s.update(status='failed',error=traceback.format_exc());write(ROOT/'controller_status.json',s);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--verify',action='store_true');p.add_argument('--train',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.verify:train(True)
    elif a.train:train()
    else:controller()
