"""Check batch4 P2-Full backward with the four most densely annotated train images."""
import torch
from common import CONFIGS, DATA, RECIPE, REPORT, RUNS, now, write
from train import PairedTrainer


def main():
    trainer=PairedTrainer(overrides={**RECIPE,'model':str(CONFIGS['full']),'data':str(DATA),
        'project':str(RUNS),'name':'dense_memory_probe','exist_ok':True,'plots':False})
    trainer.arm='full'
    trainer._setup_train(world_size=1)
    trainer.epoch=0
    trainer.batches_in_epoch=0
    dataset=trainer.train_loader.dataset
    indices=sorted(range(len(dataset)),key=lambda i:len(dataset.labels[i]['cls']),reverse=True)[:4]
    raw=dataset.collate_fn([dataset[i] for i in indices])
    batch=trainer.preprocess_batch(raw)
    torch.cuda.reset_peak_memory_stats()
    loss,items=trainer.model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    gradients=[p.grad for p in trainer.model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    write(REPORT/'dense_memory_probe.json',{'status':'passed','checked_utc':now(),
          'batch':4,'input':[4,3,640,640],'source_images':[dataset.im_files[i] for i in indices],
          'source_label_counts':[len(dataset.labels[i]['cls']) for i in indices],
          'augmented_instances':int(len(batch['cls'])),
          'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
          'loss_items':items.detach().cpu().tolist(),
          'scope':'One augmented dense batch forward/backward, no optimizer update; not an exhaustive OOM or hardware stress guarantee.'})


if __name__=='__main__':
    main()
