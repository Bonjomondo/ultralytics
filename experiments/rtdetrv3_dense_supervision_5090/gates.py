"""Real-model assignment, shared-state, gradient and deployment checks."""
from common import *
import torch,io
from ds_model import AuxModel,DenseHead,dense_loss,assign,strip_auxiliary
from ultralytics.nn.tasks import RTDETRDetectionModel
torch.set_num_threads(4)

def batch():
    return dict(img=torch.rand(2,3,128,128),batch_idx=torch.tensor([0.,0.,1.]),cls=torch.tensor([[0.],[2.],[1.]]),bboxes=torch.tensor([[.4,.4,.25,.25],[.7,.7,.15,.2],[.5,.5,.4,.3]]))

def assignment():
    h=DenseHead(nc=3);features=[torch.randn(2,256,x,x,requires_grad=True) for x in (16,8,4)];data=batch()
    for epoch in [0,30]:
        parts,d=dense_loss(h(features),data,epoch,diagnostics=True)
        assert all(torch.isfinite(x) and x>=0 for x in parts)
        assert d['positives']>0 and sum(d['positives_by_class'])==d['positives']==sum(d['positives_per_image'])
    empty=dict(data,batch_idx=torch.empty(0),cls=torch.empty(0,1),bboxes=torch.empty(0,4))
    parts,d=dense_loss(h(features),empty,0,diagnostics=True)
    assert d['positives']==0 and parts[1]==parts[2]==0
    yy,xx=torch.meshgrid(torch.arange(5)+.5,torch.arange(5)+.5,indexing='ij')
    points=torch.stack((xx,yy),-1).reshape(-1,2);gt=torch.tensor([[0.,0.,5.,5.]])
    scores,boxes,ids=assign(torch.full((25,2),.9),gt.repeat(25,1),points,torch.ones(25,1),[25],torch.tensor([1]),gt,False)
    assert (ids>=0).sum()==13 and not scores.requires_grad

def real_model(variant):
    cfg=str(ROOT/'configs'/f'{variant}.yaml')
    torch.manual_seed(0);native=RTDETRDetectionModel(cfg,nc=10,verbose=False);native.nc=10
    torch.manual_seed(0);model=AuxModel(cfg,nc=10,seed=0)
    assert digest(native.model.state_dict())==digest(model.model.state_dict())
    model.args={};data=batch();native.train();model.train();model.aux_weight=0
    torch.manual_seed(123);expected,items=native(data)
    torch.manual_seed(123);actual,parts=model(data)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(parts[:3],items,rtol=0,atol=0)
    model.aux_weight=1
    opt=torch.optim.SGD(model.parameters(),lr=.001)
    loss,_=model(data);loss.backward();opt.step();opt.zero_grad()
    for epoch in [0,30]:
        model.train_epoch=epoch;model.capture_diagnostics=True
        loss,items=model(data)
        assert torch.isfinite(loss) and items.shape==(6,)
        assert all(v['aux_norm']>0 and v['main_norm']>0 for v in model.latest_diagnostics['encoder_feature_gradients'].values())
        loss.backward();opt.zero_grad()
    model.eval();model.half().float();model.capture_diagnostics=False
    def forbid(*args):raise AssertionError('Auxiliary head called during inference')
    hook=model.aux_head.register_forward_pre_hook(forbid)
    with torch.no_grad():pred=model(data['img'])[0]
    hook.remove();deployed=strip_auxiliary(model)
    assert not any('aux' in x for x in deployed.state_dict())
    with torch.no_grad():torch.testing.assert_close(pred,deployed(data['img'])[0],rtol=0,atol=0)
    stream=io.BytesIO();torch.save(model,stream);stream.seek(0);loaded=torch.load(stream,weights_only=False)
    assert digest(loaded.state_dict())==digest(model.state_dict())
    with torch.no_grad():torch.testing.assert_close(pred,loaded(data['img'])[0],rtol=0,atol=0)

if __name__=='__main__':
    verify_pin();environment();assignment();print('assignment/static/dynamic/empty/diagnostics: passed',flush=True)
    for name in ['c0','c2']:
        real_model(name);print(name+': identity, alpha0, encoder gradients, inference removal, reload: passed',flush=True)
    write(RUNS/'gates.json',dict(status='passed',tests=['assignment/static/dynamic/empty','C0/C2 exact main init','alpha0 main loss equality','ATSS/TAL encoder gradients','no inference auxiliary','exact native strip and serialization']))
