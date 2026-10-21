"""Negative-path contract checks using the actual trained R1 model on CUDA."""
import argparse
from pathlib import Path
import torch
from detrscope.adapters import UltralyticsRTDETRAdapter
from detrscope.adapters.base import UnsupportedCapability
from detrscope.adapters.sodetr_data import CocoTargets
from detrscope.core.io import write_json, read_json, state_sha256
from detrscope.core.identity import assert_outputs_equal, snapshot_output
from detrscope.observers import StageRecorder
from detrscope.interventions import InterventionSpec
from run_validation import batches, hook_inventory
from source_adapter import PinnedR18NAAdapter


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);a=p.parse_args();repo=Path(a.repo)
    tracked=Path(__file__).resolve().parent;run=repo/'runs/detect/r18_na_ablation/rtdetr-r18-r1-150e-s0'
    ck=torch.load(run/'weights/best.pt',map_location='cpu');model=ck['model'].float().cuda().eval()
    adapter=PinnedR18NAAdapter(model,repo=repo,pin=tracked/'source_pin.json');results={}
    invalid=read_json(tracked/'source_pin.json')
    invalid['source_sha256'][next(iter(invalid['source_sha256']))]='0'*64
    invalid_path=run/'detrscope_20261021/contract_invalid_pin.json'
    write_json(invalid_path,invalid)
    try:PinnedR18NAAdapter(model,repo=repo,pin=invalid_path)
    except UnsupportedCapability as e:results['modified_source_pin_rejected']=str(e)
    else:raise AssertionError('Modified source pin accepted')
    try: UltralyticsRTDETRAdapter(model)
    except UnsupportedCapability as e: results['wheel_adapter_rejects_local_fork']=str(e)
    else: raise AssertionError('Wheel source identity gate was bypassed')
    try:adapter.require('reference_intervention')
    except UnsupportedCapability as e:results['unsupported_reference_intervention_error']=str(e)
    else:raise AssertionError('Unsupported capability was accepted')
    gt=Path('/home/zjnu/文档/ZHJ_WorkSpace/datasets/VisDrone2019-DET-YOLO/annotations/VisDrone2019-DET_val_coco.json')
    image=Path('/home/zjnu/文档/ZHJ_WorkSpace/datasets/VisDrone2019-DET-YOLO/VisDrone2019-DET-val/images')
    coco=CocoTargets(gt,model.names);items=sorted(read_json(gt)['images'],key=lambda r:r['id'])[:2]
    batch=next(batches([image/Path(r['file_name']).name for r in items],coco,'cuda:0',2))
    hooks=hook_inventory(model);weights=state_sha256(model)
    class ExpectedFailure(Exception):pass
    with torch.inference_mode():
        native=snapshot_output(model(batch.inputs))
        recorder=StageRecorder();recorder.begin_batch(batch);recorder.requested_probes={'attention','deformable_sampling'}
        try:
            with adapter.observe(recorder):
                model(batch.inputs)
                raise ExpectedFailure()
        except ExpectedFailure:pass
        assert hook_inventory(model)==hooks
        assert_outputs_equal(native,model(batch.inputs))
        results['observation_exception_cleanup_exact']=True
        try:
            with adapter.intervene(InterventionSpec(target='bbox_delta',layer_id=3,scale=0.0)):
                model(batch.inputs)
                raise ExpectedFailure()
        except ExpectedFailure:pass
        assert hook_inventory(model)==hooks
        assert_outputs_equal(native,model(batch.inputs))
        assert weights==state_sha256(model)
        results['intervention_exception_cleanup_exact']=True
    results['device']=torch.cuda.get_device_name(0);results['torch']=str(torch.__version__)
    write_json(tracked/'contract_checks.json',results);print(results)

if __name__=='__main__':main()
