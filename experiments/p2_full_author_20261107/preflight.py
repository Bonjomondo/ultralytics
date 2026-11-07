"""Verify architecture, shared initialization, stride priors, and finite gradients."""
import gc
import json
import torch

from common import ARMS, CONFIGS, RECIPE, REPORT, RUNS, SEED, dataset_identity, now, write
from models import paired, AuthorDecoder


def main():
    torch.set_num_threads(8)
    RUNS.mkdir(parents=True, exist_ok=True)
    write(REPORT / 'dataset_identity.json', dataset_identity())
    result = {'status':'running', 'seed':SEED, 'recipe':RECIPE, 'models':{}, 'checked_utc':now()}
    expected_backbone = None
    for arm in ARMS:
        model = paired(arm, SEED)
        receipt = model.initialization_receipt
        if expected_backbone is None:
            expected_backbone = receipt['backbone_sha256']
            torch.save(model.state_dict(), RUNS / 'initialization_b0_s1.pt')
        assert receipt['backbone_sha256'] == expected_backbone
        model = model.cuda().eval()
        head = model.model[-1]
        shapes = []
        handle = head.register_forward_pre_hook(lambda module, args: shapes.extend([list(x.shape) for x in args[0]]))
        with torch.inference_mode():
            output = model(torch.rand(1, 3, 640, 640, device='cuda'))[0]
        handle.remove()
        expected = [4,8,16,32] if arm == 'full' else [8,16,32]
        assert [640 // shape[-1] for shape in shapes] == expected
        assert list(output.shape) == [1,300,14] and torch.isfinite(output).all()
        anchors, mask = head._generate_anchors([shape[-2:] for shape in shapes])
        reference, reference_mask = AuthorDecoder._generate_anchors(head, [[80,80],[40,40],[20,20]])
        start = 25600 if arm == 'full' else 0
        assert torch.equal(anchors[:,start:], reference)
        assert torch.equal(mask[:,start:], reference_mask)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        targets = {'gt_groups':[1]*4, 'cls':torch.zeros(4,device='cuda',dtype=torch.long),
                   'bboxes':torch.tensor([[.5,.5,.1,.1]]*4,device='cuda'),
                   'batch_idx':torch.arange(4,device='cuda')}
        outputs = model.predict(torch.rand(4,3,640,640,device='cuda'), batch=targets)
        loss = sum(x.float().square().mean() for x in outputs[:4])
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        if arm != 'b0':
            gradient = model.model[22 if arm == 'full' else 17].conv.weight.grad
            assert gradient is not None and gradient.abs().sum() > 0
        result['models'][arm] = {
            'parameters':sum(p.numel() for p in model.parameters()), 'feature_shapes':shapes,
            'feature_tokens':sum(shape[-1]*shape[-2] for shape in shapes),
            'strides':expected, 'output_shape':list(output.shape), 'finite_forward_backward':True,
            'p3_p4_p5_anchors_preserved':True,
            'synthetic_batch4_peak_gib':torch.cuda.max_memory_allocated()/2**30,
            'initialization':receipt}
        del model, outputs, loss, grads, output
        gc.collect();torch.cuda.empty_cache()
    result['status'] = 'passed'
    write(REPORT / 'preflight.json', result)
    write(REPORT / 'experiment_spec.json', {
        'arms':ARMS, 'recipe':RECIPE, 'seed':SEED, 'milestones':[150,250,350],
        'checkpoints_per_milestone':['last','best'], 'fresh_initialization':True,
        'backbone_sha256':expected_backbone,
        'backend':'tracked SO-DETR-derived Ultralytics8.0.201 from c2_channel_author_20261105',
        'anchor_wh_normalized':{'P2':.025,'P3':.05,'P4':.1,'P5':.2},
        'formal_evaluation':'converted VisDrone val548 COCOeval maxDets100, maxDets300 supplement',
        'tiny_bins':'short side in square-resized640: [0,8),[8,16),[16,32),[32,inf)',
        'confirmation':'No automatic extra seeds or extra architecture ablations in this queue.'})
    print(json.dumps({arm:{k:v for k,v in record.items() if k != 'initialization'}
                      for arm,record in result['models'].items()},indent=2))


if __name__ == '__main__':
    main()
