"""Check author-engine C2 identity and the B1 initialization receipt."""
from pathlib import Path
from types import SimpleNamespace
import json
import torch
from ultralytics.nn.tasks import RTDETRDetectionModel
from train_scalar import PairedTrainer, state_hash

ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(8)
    t=object.__new__(PairedTrainer)
    t.data={'nc':10};t.args=SimpleNamespace(seed=0,resume=False);t.save_dir=ROOT/'preflight'
    t.save_dir.mkdir(exist_ok=True)
    torch.manual_seed(0)
    m=t.get_model(str(ROOT/'rtdetr-resnet18-scalar.yaml'),verbose=False)
    info=json.loads((t.save_dir/'initialization.json').read_text())
    old=Path('/home/zjnu/文档/ZHJ_WorkSpace/reproductions/rtdetr-r18-c2-author-recipe-20261028')
    receipt=json.loads((old/'runs/rtdetr-r18-c2-matched-b0-350e-s0/initialization.json').read_text())
    assert info['common_state_sha256']==receipt['common_state_sha256']
    torch.manual_seed(0)
    ref=RTDETRDetectionModel(str(ROOT/'rtdetr-resnet18-c2.yaml'),nc=10,verbose=False)
    ref.load_state_dict({k:v for k,v in m.state_dict().items() if k in ref.state_dict()},strict=True)
    m=m.cuda().eval();ref=ref.cuda().eval();x=torch.rand(1,3,640,640,device='cuda')
    with torch.inference_mode():
        error=float((m(x)[0]-ref(x)[0]).abs().max())
    assert error==0
    gate=m.model[18];xs=[torch.rand(2,c,8,8,device='cuda',requires_grad=True) for c in (256,128,128)]
    gate(xs).square().mean().backward()
    assert gate.logit.grad.abs()>0 and xs[1].grad.abs().sum()>0
    result={'status':'passed','common_initialization_matches_B1_receipt':True,'whole_model_640_max_abs_error_vs_C2':error,
            'gate_and_detail_gradient_pass':True,'parameters':sum(p.numel() for p in m.parameters()),'initial_multiplier':1.0}
    (ROOT/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

if __name__=='__main__':main()
