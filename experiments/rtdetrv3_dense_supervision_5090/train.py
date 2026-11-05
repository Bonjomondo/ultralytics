"""Paired seed0 author-recipe training, stopped at the 150-epoch screening gate."""
from common import *
import argparse,time,traceback,shutil
from datetime import datetime,timezone
from copy import copy
import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ds_model import AuxModel

class Trainer(RTDETRTrainer):
    arm=None
    def get_model(self,cfg=None,weights=None,verbose=True):
        assert weights is None and not self.args.resume,'Fresh runs only; recovery requires exact checkpoint audit'
        is_c2=self.arm.startswith('a3')
        aux=self.arm.startswith(('a2','a3'))
        model=AuxModel(cfg,nc=10,seed=0,verbose=False) if aux else RTDETRDetectionModel(cfg,nc=10,verbose=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            reference=RTDETRDetectionModel(str(ROOT/'configs/c0.yaml'),nc=10,verbose=False)
        if is_c2:
            a,b=reference.state_dict(),model.state_dict()
            common=[k for k,v in b.items() if not k.startswith(('model.17.','aux_head.')) and k in a and v.shape==a[k].shape]
            with torch.no_grad():
                for k in common:b[k].copy_(a[k])
            assert digest(b,common)==digest(a,common)=='cd1cf17318a869866d8bdf80e67e76a16828c783e8361c23b82c1e579ce3a7c0'
        else:
            common=list(reference.state_dict())
            assert digest(model.state_dict(),common)==digest(reference.state_dict())
        write(self.save_dir/'initialization.json',dict(seed=0,aux_seed=4000 if aux else None,main_state_sha256=digest(model.model.state_dict()),common_sha256=digest(model.state_dict(),common),common_tensors=len(common),trained_checkpoint_loaded=False,method='Author C0 seed0 constructor; C2 copies 527 common tensors excluding changed layer17; auxiliary initialized inside isolated RNG fork.'))
        return model

    def get_validator(self):
        from ultralytics.models.rtdetr.val import RTDETRValidator
        self.loss_names=('giou_loss','cls_loss','l1_loss','aux_vfl','aux_giou','aux_dfl') if self.arm.startswith(('a2','a3')) else ('giou_loss','cls_loss','l1_loss')
        return RTDETRValidator(self.test_loader,save_dir=self.save_dir,args=copy(self.args))

    def final_eval(self):
        # Preserve complete optimizer, EMA and best/last; formal eval is separate.
        assert self.last.exists()

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['a0_c0','a2_c0_o2mc','a3_c2_o2mc']);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    verify_pin();env=environment();torch.set_num_threads(8)
    from ultralytics.utils import callbacks
    callbacks.add_integration_callbacks=lambda instance:None
    r=recipe(a.arm,a.smoke);run=Path(r['project'])/a.arm
    assert not run.exists(),f'Will not overwrite {run}'
    Trainer.arm=a.arm
    t=Trainer(overrides=r)
    state=dict(status='starting',arm=a.arm,pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),environment=env,recipe=r,completed_epochs=0)
    write(run/'receipt.json',state);write(run/'source_manifest.json',source_hashes())
    begin=time.time();batch_count=0
    def epoch_start(tr):
        nonlocal batch_count
        batch_count=0
        tr.model.train_epoch=tr.epoch
    def before(tr):
        nonlocal batch_count
        batch_count+=1
        if isinstance(tr.model,AuxModel):tr.model.capture_diagnostics=batch_count==3
    def after(tr):
        if batch_count==3 and isinstance(tr.model,AuxModel):
            d=tr.model.latest_diagnostics
            if d is not None:
                with (run/'aux_diagnostics.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=tr.epoch+1,batch=batch_count,**d))+'\n')
        if batch_count==1 or batch_count%100==0:
            assert torch.isfinite(tr.tloss).all(),tr.tloss
            state.update(status='training',epoch=tr.epoch+1,batch=batch_count,updated_utc=datetime.now(timezone.utc).isoformat(),loss=tr.tloss.tolist())
            write(run/'receipt.json',state)
    def saved(tr):
        if tr.epoch+1==(1 if a.smoke else 150):
            for name,path in [('best',tr.best),('last',tr.last)]:
                shutil.copy2(path,tr.wdir/f'epoch{tr.epoch+1:03d}_{name}.pt')
    def finished(tr):
        state.update(status='training',completed_epochs=tr.epoch+1,elapsed_seconds=time.time()-begin,epoch_seconds=tr.epoch_time,gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,accumulate=tr.accumulate,internal_metrics=tr.metrics,loss=tr.tloss.tolist(),updated_utc=datetime.now(timezone.utc).isoformat())
        assert state['accumulate']==1,state['accumulate']
        with (run/'health.jsonl').open('a') as f:f.write(json.dumps(state,default=str)+'\n')
        write(run/'receipt.json',state)
        if not a.smoke and tr.epoch+1>=150:tr.stop=True
    for event,fn in [('on_train_epoch_start',epoch_start),('on_train_batch_start',before),('on_train_batch_end',after),('on_model_save',saved),('on_fit_epoch_end',finished)]:t.add_callback(event,fn)
    try:
        t.train()
        assert state['completed_epochs']==(1 if a.smoke else 150),f'Early stop before fixed screening budget: {state}'
        state.update(status='completed_150_pending_formal_evaluation' if not a.smoke else 'smoke_completed',finished_utc=datetime.now(timezone.utc).isoformat(),checkpoints={label:dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)) for label,path in [('best',t.best),('last',t.last)]})
        write(run/'receipt.json',state)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());write(run/'receipt.json',state);raise
if __name__=='__main__':main()
