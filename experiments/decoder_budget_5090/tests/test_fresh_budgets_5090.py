"""Fresh initialization and physical training architecture, including one decoder layer."""
import json
import argparse
from pathlib import Path

import pytest
import torch

from experiments.decoder_budget_5090 import train_fresh_budgets_5090 as train
from experiments.decoder_budget_5090 import report_fresh_budgets_5090 as report
from experiments.decoder_budget_5090 import queue_fresh_budgets_5090 as queue


@pytest.mark.parametrize("variant",list(train.BUDGETS))
def test_all_four_architectures_train_with_real_loss_and_gradients(variant):
    torch.set_num_threads(4)
    model=train.fresh_model(train.CONFIGS[variant])
    q,p,d=train.BUDGETS[variant]
    architecture=train.inspect_architecture(model,variant)
    assert architecture["decoder_layers"]==d
    assert len(model.model[-1].decoder.layers)==d
    batch={"img":torch.rand(2,3,128,128),"cls":torch.tensor([[0.],[1.]]),
           "bboxes":torch.tensor([[.5,.5,.2,.2],[.3,.4,.15,.2]]),"batch_idx":torch.tensor([0,1])}
    model.train()
    loss,components=model(batch)
    assert torch.isfinite(loss) and torch.isfinite(components).all()
    loss.backward()
    assert next(model.model[14].parameters()).grad.abs().sum()>0
    for layer in model.model[-1].decoder.layers:
        assert layer.cross_attn.n_points==p
        assert layer.cross_attn.sampling_offsets.weight.grad.abs().sum()>0
    assert model.ablation_initialization["pretrained"] is False
    assert model.ablation_initialization["source_checkpoint"] is None


def test_existing_weights_are_refused_and_queries_do_not_change_fresh_parameter_values():
    with pytest.raises(ValueError,match="forbids"):
        train.fresh_model(train.CONFIGS['c2-q300-p2-d3'],weights=object())
    a=train.fresh_model(train.CONFIGS['c2-q300-p2-d3'])
    b=train.fresh_model(train.CONFIGS['c2-q200-p2-d3'])
    assert a.ablation_initialization['full_initial_state_sha256']==b.ablation_initialization['full_initial_state_sha256']
    assert a.model[-1].num_queries==300 and b.model[-1].num_queries==200


def test_all_four_share_identical_common_fresh_tensors():
    plan=train.initialization_plan()
    assert plan['checkpoint_initialization'] is False
    assert len(plan['common_keys'])>200
    assert len({r['initialization']['reference_c2_full_sha256'] for r in plan['variants'].values()})==1
    assert len({r['initialization']['reference_r0_shared_sha256'] for r in plan['variants'].values()})==1
    assert plan['variants']['c2-q300-p4-d1']['architecture']['parameters']<plan['variants']['c2-q300-p4-d2']['architecture']['parameters']


def test_report_replays_without_training_data_or_checkpoints(tmp_path):
    data={'schema':'5090.fresh-budget-report.v1','generated_utc':'fixed','root':'/server-only',
          'queue':{'status':'gates_passed'},'records':[]}
    report.generate(tmp_path/'absent',tmp_path/'report',data=data)
    assert json.loads((tmp_path/'report/experiments.json').read_text())==data
    assert 'gates_passed' in (tmp_path/'report/experiments.md').read_text()


def test_an_existing_training_directory_is_never_reused(tmp_path):
    label=next(iter(train.BUDGETS))
    project=tmp_path/'5090-owned';run=project/'5090-run';run.mkdir(parents=True)
    sentinel=run/'keep.txt';sentinel.write_text('original')
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'variants':{label:{'config_sha256':train.sha256_file(train.CONFIGS[label])}}}))
    args=argparse.Namespace(variant=label,epochs=150,fraction=1.,project=project,name='5090-run',data=tmp_path/'data.yaml',plan=plan)
    with pytest.raises(FileExistsError):train.train(args)
    assert sentinel.read_text()=='original' and list(run.iterdir())==[sentinel]


def mock_queue(tmp_path,monkeypatch):
    args=argparse.Namespace(root=tmp_path/'5090-queue',reports=tmp_path/'reports',continue_after_gates=True,gate_only=False)
    q=queue.Queue(args)
    q.state={'arms':{v:{'status':'pending'} for v in train.BUDGETS},'phases':[]};q.owned=True
    monkeypatch.setattr(q,'prepare',lambda:None)
    monkeypatch.setattr(q,'update',lambda **kwargs:q.state.update(kwargs))
    return q


def test_one_arm_failure_does_not_prevent_the_other_three_requested_trainings(tmp_path,monkeypatch):
    q=mock_queue(tmp_path,monkeypatch);seen=[]
    def training(v,gate):
        seen.append(v)
        if len(seen)==1:raise RuntimeError('simulated arm failure')
        return tmp_path/v
    monkeypatch.setattr(q,'training',training)
    monkeypatch.setattr(q,'evaluation',lambda *a:{'metrics':{},'benchmark':{'status':'completed'}})
    q.run()
    assert seen==list(train.BUDGETS)
    assert q.state['status']=='completed_with_failures'
    assert sum(a['status']=='completed' for a in q.state['arms'].values())==3


def test_source_change_stops_the_whole_queue_instead_of_training_unknown_code(tmp_path,monkeypatch):
    q=mock_queue(tmp_path,monkeypatch);seen=[]
    def training(v,gate):seen.append(v);raise queue.SourceChanged('source changed')
    monkeypatch.setattr(q,'training',training)
    with pytest.raises(queue.SourceChanged):q.run()
    assert len(seen)==1 and q.state['status']=='failed'
