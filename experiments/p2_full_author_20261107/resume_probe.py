"""Verify an actual interrupted two-epoch short run resumes raw state and advances."""
import json
from common import CONFIGS, RECIPE, REPORT, RUNS, sha, write
from train import PairedTrainer


def trainer(recipe):
    instance = PairedTrainer(overrides=recipe)
    instance.arm = 'full'
    for event, method in [('on_train_start',instance.attach_diagnostics),
                          ('on_train_epoch_start',instance.epoch_start),
                          ('on_train_batch_start',instance.batch_start),
                          ('on_train_batch_end',instance.batch_end),
                          ('on_fit_epoch_end',instance.fit_end)]:
        instance.add_callback(event,method)
    return instance


def main():
    recipe={**RECIPE,'epochs':2,'model':str(CONFIGS['full']),'data':str(RUNS/'smoke.yaml'),
            'project':str(RUNS),'name':'resume_probe_full','exist_ok':True,'plots':False}
    checkpoint=RUNS/'resume_probe_full/weights/last.pt'
    assert not checkpoint.exists(),'Do not overwrite a previous probe'
    first=trainer(recipe)
    first.add_callback('on_fit_epoch_end',lambda t:setattr(t,'stop',True) if t.epoch==0 else None)
    first.train()
    assert first.epoch==0
    initial_sha=sha(checkpoint)
    del first
    second=trainer({**recipe,'model':str(checkpoint),'resume':str(checkpoint)})
    second.train()
    assert second.epoch==1
    receipt=json.loads((checkpoint.parents[1]/'restore_epoch1.json').read_text())
    assert receipt['checkpoint_sha256']==initial_sha
    assert all(receipt[k] for k in ['raw_exact','ema_exact','optimizer_exact'])
    write(REPORT/'resume_verification.json',{'status':'passed','completed_epochs':2,
          'receipt':receipt,'final_checkpoint_sha256':sha(checkpoint),
          'scope':'Actual two-epoch smoke training, stopped at epoch1 and raw-resumed to epoch2.'})


if __name__=='__main__':
    main()
