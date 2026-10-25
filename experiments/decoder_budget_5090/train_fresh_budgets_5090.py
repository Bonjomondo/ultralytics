"""Fresh C2 decoder-budget training. No pretrained or existing checkpoint input.

The reference is generated in RAM from seed0, not loaded from disk. Copying its
matching random tensors keeps the historical C2 initialization on common layers.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import torch
import yaml

from experiments.decoder_budget_5090.train_5090 import (
    ROOT, MAP_KEY, finite_numbers, git, read_health, read_rows, runtime_environment,
    sha256_file, write_atomic,
)
from experiments.r18_p2_sfi.train import MODELS, RECIPE, build_model as build_reference, tensor_digest
from train_rtdetr_r18_na import TrainingHealth
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import unwrap_model

HERE = Path(__file__).resolve().parent
BUDGETS = {f"c2-q{q}-p{p}-d{d}": (q,p,d) for q,p,d in ((300,2,3),(200,2,3),(300,4,2),(300,4,1))}
CONFIGS = {key: HERE/"configs"/(key+".yaml") for key in BUDGETS}


def now():
    return datetime.now(timezone.utc).isoformat()


def budget_config(cfg):
    data = yaml.safe_load(Path(cfg).read_text()) if isinstance(cfg,(str,Path)) else cfg
    label = data.get("decoder_budget",{}).get("variant")
    if label not in BUDGETS:
        raise ValueError("Unknown scratch-training budget")
    q,p,d = BUDGETS[label]
    if data["decoder_budget"] != {"variant":label,"queries":q,"points":p,"layers":d}:
        raise ValueError("Budget metadata differs from requested architecture")
    if data["head"][-1][2:] != ["RTDETRDecoder",["nc",256,q,p,8,d]]:
        raise ValueError("YAML must construct the requested query/point/layer dimensions")
    return data,label


def fresh_model(cfg,nc=10,ch=3,seed=0,weights=None,verbose=False):
    if weights is not None:
        raise ValueError("Fresh training forbids checkpoint/pretrained transfer or resume")
    data,label = budget_config(cfg)
    with torch.random.fork_rng(devices=[]):
        reference = build_reference(str(MODELS["c2"]),nc=nc,ch=ch,seed=seed)
        torch.manual_seed(seed+1000)
        model = RTDETRDetectionModel(data,nc=nc,ch=ch,verbose=verbose)
        model.nc=nc  # also support direct loss/backward checks before trainer attributes are attached
        src,dst = reference.state_dict(),model.state_dict()
        common = {k:v for k,v in src.items() if k in dst and v.shape==dst[k].shape}
        model.load_state_dict(common,strict=False)
        actual = model.state_dict()
        if any(not torch.equal(actual[k],v) for k,v in common.items()):
            raise RuntimeError("Fresh shared initialization mismatch")
        model.ablation_initialization = {
            "seed":seed,"candidate_seed":seed+1000,"variant":label,
            "source":"fresh seeded C2 generated in RAM; no checkpoint/backbone pretrained weights read",
            "pretrained":False,"resume":False,"source_checkpoint":None,
            "reference_c2_full_sha256":tensor_digest(src),
            "reference_r0_shared_sha256":reference.ablation_initialization["shared_state_sha256"],
            "shared_state_sha256":tensor_digest(common),"shared_state_items":len(common),"shared_state_verified":True,
            "full_initial_state_sha256":tensor_digest(actual),
            "shape_changed_state":{k:[list(src[k].shape),list(v.shape)] for k,v in actual.items() if k in src and src[k].shape!=v.shape},
            "removed_reference_state":sorted(set(src)-set(actual)),
            "new_native_initialized_state":sorted(set(actual)-set(common)),
        }
    return model


def inspect_architecture(model,label):
    q,p,d = BUDGETS[label]
    h = model.model[-1]
    if (h.num_queries,h.decoder.num_layers,h.decoder.eval_idx)!=(q,d,d-1):
        raise RuntimeError("Native query count/physical depth does not match configuration")
    if len(h.decoder.layers)!=d or len(h.dec_score_head)!=d or len(h.dec_bbox_head)!=d:
        raise RuntimeError("Unused decoder layers/heads remain in the architecture")
    for layer in h.decoder.layers:
        a=layer.cross_attn
        if a.n_points!=p or a.sampling_offsets.out_features!=8*3*p*2 or a.attention_weights.out_features!=8*3*p:
            raise RuntimeError("Sampling dimensions do not match requested training model")
    return {"queries":q,"points_per_level":[p]*3,"decoder_layers":d,
            "parameters":sum(v.numel() for v in model.parameters()),"eval_idx":d-1}


def initialization_plan():
    records,states = {},{}
    for label,cfg in CONFIGS.items():
        model = fresh_model(cfg)
        records[label] = {"config":str(cfg),"config_sha256":sha256_file(cfg),
                          "architecture":inspect_architecture(model,label),"initialization":model.ablation_initialization}
        states[label] = {k:v.detach().clone() for k,v in model.state_dict().items()}
    keys = set.intersection(*(set(v) for v in states.values()))
    keys = sorted(k for k in keys if len({tuple(v[k].shape) for v in states.values()})==1)
    reference = next(iter(states.values()))
    if any(not torch.equal(v[k],reference[k]) for v in states.values() for k in keys):
        raise RuntimeError("Shared tensors across the four fresh architectures differ")
    return {"schema":"5090.fresh-budget-init.v1","seed":0,"checkpoint_initialization":False,
            "common_keys":keys,"common_state_sha256":tensor_digest({k:reference[k] for k in keys}),
            "variants":records}


class FreshBudgetTrainer(RTDETRTrainer):
    def get_model(self,cfg=None,weights=None,verbose=True):
        return fresh_model(cfg,self.data["nc"],self.data["channels"],self.args.seed,weights,verbose)

    def final_eval(self):
        if self.last.exists():
            shutil.copy2(self.last,self.wdir/"resume.pt")
        super().final_eval()


class BudgetHealth(TrainingHealth):
    """Dynamic depth/queries; D1 steps compare with encoder, not a nonexistent D0."""
    def start(self,trainer):
        if trainer.batch_size!=8 or trainer.amp:
            raise RuntimeError("Require batch8 FP32")
        self.refine=None
        model=unwrap_model(trainer.model);h=model.model[-1]
        for label,mlp in [("encoder",h.enc_bbox_head),*[(f"d{i+1}",m) for i,m in enumerate(h.dec_bbox_head)]]:
            def hidden(module,args,label=label):
                if module.training and self.capture:
                    value=args[0][:,-h.num_queries:].float()
                    self._sample(label+"_hidden_active_ratio",(value>0).float().mean())
                    self._sample(label+"_hidden_query_variance",value.var(1,unbiased=False).mean())
            def delta(module,args,out,label=label):
                if module.training and self.capture:
                    value=out[:,-h.num_queries:].float()
                    self._sample(label+"_delta_abs",value.abs().mean())
                    self._sample(label+"_delta_finite",torch.isfinite(value).float().mean())
                    self._sample(label+"_delta_query_variance",value.var(1,unbiased=False).mean())
            self.handles += [mlp.layers[-1].register_forward_pre_hook(hidden),mlp.register_forward_hook(delta)]
        def step(module,args,out):
            if module.training and self.capture:
                boxes=out[0][:,:,-h.num_queries:]
                previous=boxes[-2] if len(boxes)>1 else out[2]
                self._sample("last_bbox_step_abs",(boxes[-1]-previous).float().abs().mean())
        self.handles.append(h.register_forward_hook(step))
        self.handles.append(next(model.model[14].parameters()).register_hook(
            lambda grad:self._sample("p2_first_weight_grad_abs",grad.float().abs().mean())))
        for i,layer in enumerate(h.decoder.layers):
            self.handles.append(layer.cross_attn.sampling_offsets.weight.register_hook(
                lambda grad,i=i:self._sample(f"d{i+1}_offset_weight_grad_abs",grad.float().abs().mean())))
        record={**model.ablation_initialization,"model_config":str(trainer.args.model),"amp":trainer.amp,
                "gradient_sampling":"first three training batches per epoch; normal matching queries only for activations"}
        (trainer.save_dir/"initialization.json").write_text(json.dumps(record,indent=2)+"\n")

    def batch_end(self,trainer):
        if trainer.batch_size!=8 or not torch.isfinite(trainer.loss).all():
            raise FloatingPointError("Changed batch size or nonfinite training loss")
        super().batch_end(trainer)


def frozen_sources():
    paths=[*ROOT.joinpath("ultralytics").rglob("*.py"),*CONFIGS.values(),MODELS["c0"],MODELS["c2"],
           ROOT/"train_rtdetr_r18_na.py",ROOT/"train_paper_repro_visdrone.py",
           ROOT/"experiments/r18_p2_sfi/train.py",HERE/"train_5090.py",
           HERE/"train_fresh_budgets_5090.py",HERE/"queue_fresh_budgets_5090.py",
           HERE/"evaluate_fresh_budgets_5090.py",HERE/"report_fresh_budgets_5090.py"]
    paths += [HERE/name for name in ("run.py","screen_decoder_perf_5090.py","profile_runtime_5090.py")]
    return {str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(set(paths))}


def train(args):
    if args.variant not in BUDGETS or args.epochs<1 or not 0<args.fraction<=1:
        raise ValueError("Invalid variant/epoch/fraction")
    args.project=args.project.resolve();args.data=args.data.resolve()
    if "5090" not in args.project.name or not args.name.startswith("5090-") or Path(args.name).name!=args.name:
        raise ValueError("Use an owned 5090 run namespace")
    plan=json.loads(args.plan.read_text())
    expected=plan["variants"][args.variant]
    if sha256_file(CONFIGS[args.variant])!=expected["config_sha256"]:
        raise RuntimeError("Configuration differs from initialization plan")
    run=args.project/args.name
    run.mkdir(parents=True,exist_ok=False)
    state={"schema":"5090.fresh-budget-training.v1","status":"running","phase":"preflight",
           "variant":args.variant,"experiment_name":args.name,"run":str(run),"seed":0,
           "epochs_requested":args.epochs,"current_epoch":0,"fraction":args.fraction,
           "role":"fresh_150e" if args.epochs==150 and args.fraction==1 else "engineering_gate",
           "started_utc":now(),"initialization_plan_sha256":sha256_file(args.plan),
           "data_yaml":str(args.data),"data_sha256":sha256_file(args.data),
           "source_commit":git("rev-parse","HEAD"),"source_branch":git("branch","--show-current"),
           "source_sha256":frozen_sources(),"pretrained":False,"resume":False,
           "checkpoint_storage":"native FP16 EMA/optimizer checkpoint storage; evaluation converts to FP32; resume.pt kept unstripped"}
    def save():
        state["updated_utc"]=now()
        write_atomic(run/"experiment.json",state)
    def refresh():
        try:
            from experiments.decoder_budget_5090.report_fresh_budgets_5090 import generate
            generate(args.project,args.reports)
            state["reporting"]={"status":"completed","output":str(args.reports)}
        except Exception as error:
            state["reporting"]={"status":"failed","error":str(error)}
    save()
    try:
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        state["environment"]=runtime_environment();save()
        if not state["environment"]["cuda_available"] or "5090" not in state["environment"].get("gpu",""):
            raise RuntimeError("CUDA RTX 5090 is required")
        wrapper=RTDETR(str(CONFIGS[args.variant]))
        BudgetHealth().attach(wrapper)
        def started(trainer):
            model=unwrap_model(trainer.model)
            architecture=inspect_architecture(model,args.variant)
            initial=json.loads((run/"initialization.json").read_text())
            if (trainer.args.pretrained or trainer.args.resume or trainer.amp or trainer.batch_size!=8
                    or trainer.args.seed!=0 or trainer.args.workers!=0 or trainer.save_dir.resolve()!=run):
                raise RuntimeError("Trainer changed the fresh FP32 batch8 seed0 recipe")
            if initial["full_initial_state_sha256"]!=expected["initialization"]["full_initial_state_sha256"]:
                raise RuntimeError("Actual trainer model differs from planned fresh initialization")
            values=model.state_dict()
            actual=tensor_digest(values)
            common=tensor_digest({k:values[k] for k in plan["common_keys"]})
            if actual!=initial["full_initial_state_sha256"] or common!=plan["common_state_sha256"]:
                raise RuntimeError("Initial model was modified or common tensors are not paired")
            state.update(phase="training",architecture=architecture,initialization=initial,environment=runtime_environment(),
                         cpu_initial_seed=torch.initial_seed(),cuda_initial_seed=torch.cuda.initial_seed(),
                         full_initial_state_verified=True,common_state_sha256=common,
                         effective_train_args={k:getattr(trainer.args,k) for k in RECIPE}|{"pretrained":False,"resume":False,"workers":0})
            save();refresh();save()
        def epoch_end(trainer):
            rows=read_rows(run/"results.csv");health=read_health(run/"health.jsonl")
            if not rows or not health:
                raise RuntimeError("Missing epoch evidence")
            finite_numbers(health[-1],location="health")
            state.update(current_epoch=len(rows),best_internal_map=max(float(r[MAP_KEY]) for r in rows),
                         latest_internal_map=float(rows[-1][MAP_KEY]),latest_health=health[-1],
                         train_seconds=sum(h["train_seconds"] for h in health))
            save();refresh();save()
        wrapper.add_callback("on_train_start",started)
        wrapper.add_callback("on_fit_epoch_end",epoch_end)
        wrapper.train(trainer=FreshBudgetTrainer,**{**RECIPE,"epochs":args.epochs},data=str(args.data),device="0",
                      workers=0,fraction=args.fraction,project=str(args.project),name=args.name,
                      pretrained=False,resume=False,exist_ok=True)
        rows=read_rows(run/"results.csv");health=read_health(run/"health.jsonl")
        if len(rows)!=args.epochs or len(health)!=args.epochs or [h["epoch"] for h in health]!=list(range(1,args.epochs+1)):
            raise RuntimeError("Incomplete requested epoch budget")
        import math
        if any(not math.isfinite(float(v)) for row in rows for v in row.values()):
            raise FloatingPointError("Nonfinite training curve")
        for h in health:
            finite_numbers(h,location="health")
            if any(v!=1 for k,v in h.items() if k.endswith("_finite")):
                raise FloatingPointError("Nonfinite bbox deltas")
        if not any((h.get("p2_first_weight_grad_abs") or 0)>0 for h in health):
            raise RuntimeError("P2 branch has no measured gradient")
        for i in range(BUDGETS[args.variant][2]):
            if not any((h.get(f"d{i+1}_offset_weight_grad_abs") or 0)>0 for h in health):
                raise RuntimeError(f"Decoder {i+1} offset projection has no measured gradient")
        if frozen_sources()!=state["source_sha256"] or sha256_file(args.data)!=state["data_sha256"]:
            raise RuntimeError("Frozen training source or dataset YAML changed")
        checkpoints={}
        for name in ("best","last","resume"):
            path=run/"weights"/(name+".pt")
            if not path.is_file():
                raise RuntimeError("Missing checkpoint: "+str(path))
            checkpoints[name]={"path":str(path),"bytes":path.stat().st_size,"sha256":sha256_file(path)}
        saved=torch.load(run/"weights/resume.pt",map_location="cpu",weights_only=False)
        if saved.get("epoch")!=args.epochs-1 or saved.get("optimizer") is None:
            raise RuntimeError("Unstripped restart checkpoint incomplete")
        state.update(status="completed",phase="done",completed_utc=now(),checkpoints=checkpoints,epochs_completed=len(rows))
        save();refresh();save()
        return state
    except BaseException as error:
        state.update(status="failed",phase="failed",error=type(error).__name__+": "+str(error))
        save();refresh();save()
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant",choices=BUDGETS,required=True)
    p.add_argument("--project",type=Path,required=True)
    p.add_argument("--name",required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--reports",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=150)
    p.add_argument("--fraction",type=float,default=1.)
    train(p.parse_args())


if __name__=="__main__":
    main()
