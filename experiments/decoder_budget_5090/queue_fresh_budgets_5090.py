"""Durable serial queue: four fresh gates, then four complete 150e runs and evaluations."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.decoder_budget_5090.train_fresh_budgets_5090 import (
    ROOT,HERE,BUDGETS,CONFIGS,frozen_sources,initialization_plan,now,
)
from experiments.decoder_budget_5090.train_5090 import sha256_file,write_atomic
from experiments.decoder_budget_5090.report_fresh_budgets_5090 import generate

DATA=ROOT.parent/"SO_DETR/datasets/VisDrone2019-DET-YOLO"


class SourceChanged(RuntimeError):
    pass


class Queue:
    def __init__(self,args):
        self.args=args;self.root=args.root.resolve();self.state={};self.child=None;self.owned=False
        self.env={**os.environ,"PYTHONPATH":str(ROOT)+":"+str(ROOT.parent/"detrscope/src"),
                  "YOLO_CONFIG_DIR":"/tmp/zhj-uav-5090-config","MPLBACKEND":"Agg","PYTHONUNBUFFERED":"1","OMP_NUM_THREADS":"4"}

    def update(self,**values):
        self.state.update(values,updated_utc=now())
        write_atomic(self.root/"status.json",self.state)
        try:generate(self.root,self.args.reports)
        except Exception as e:self.state["report_error"]=str(e)
        write_atomic(self.root/"status.json",self.state)

    def check_sources(self):
        frozen=json.loads((self.root/"frozen_sources.json").read_text())
        if frozen_sources()!=frozen["source_sha256"]:
            raise SourceChanged("Frozen experiment/model source changed")
        for path,expected in frozen["inputs"].items():
            if sha256_file(Path(path))!=expected:
                raise SourceChanged("Frozen experiment input changed: "+path)

    def prepare(self):
        if self.args.continue_after_gates:
            self.state=json.loads((self.root/"status.json").read_text())
            if self.state["status"]!="gates_passed":
                raise RuntimeError("Only a completed gate-only run can continue")
            self.check_sources();self.validate_gates();self.owned=True
            self.update(status="running",pid=os.getpid(),continued_utc=now())
            return
        self.root.mkdir(parents=True,exist_ok=False);self.owned=True
        self.state={"schema":"5090.fresh-budget-queue.v1","status":"running","started_utc":now(),"pid":os.getpid(),
                    "scope":"all four C2 budget architectures from scratch; no existing checkpoint/pretrained/finetuning",
                    "order":list(BUDGETS),"epochs_per_variant":150,"seed":0,"phases":[],
                    "arms":{k:{"status":"pending"} for k in BUDGETS},"reports":str(self.args.reports)}
        self.update(phase="initialization_plan")
        plan=initialization_plan();write_atomic(self.root/"initialization_plan.json",plan)
        import yaml
        data=yaml.safe_load((ROOT/"docs/experiment_reports/2026-10-24_5090_baselines/raw/data_5090.yaml").read_text())
        (self.root/"data.yaml").write_text(yaml.safe_dump(data,sort_keys=False,allow_unicode=True))
        script=ROOT/"docs/experiment_reports/2026-10-24_5090_baselines/audit_assets_5090.py"
        spec=importlib.util.spec_from_file_location("fresh_data_audit",script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        audit=module.audit(DATA,ROOT/"experiments/r18_na_ablation/completed_pair/r0_formal_metrics.json")
        old=json.loads((script.parent/"raw/asset_audit_5090.json").read_text())
        if audit["status"]!="passed" or any(audit[k]!=old[k] for k in ("label_content_sha256","image_inventory_sha256")) or audit["coco"]["sha256"]!=old["coco"]["sha256"]:
            raise RuntimeError("Dataset identity differs from the C0/C2 baseline")
        write_atomic(self.root/"asset_audit.json",audit)
        inputs=[self.root/"data.yaml",self.root/"initialization_plan.json",DATA/"annotations/VisDrone2019-DET_val_coco.json"]
        write_atomic(self.root/"frozen_sources.json",{"source_sha256":frozen_sources(),"inputs":{str(p):sha256_file(p) for p in inputs}})

    def phase(self,label,command,output):
        self.check_sources()
        log=self.root/"logs"/(label+".log");log.parent.mkdir(exist_ok=True)
        if output.exists() or log.exists():
            raise FileExistsError("Refusing existing phase: "+label)
        entry={"name":label,"command":command,"log":str(log),"output":str(output),"started_utc":now(),"status":"running"}
        self.state["phases"].append(entry);self.update(phase=label,child_pid=None)
        with log.open("x") as stream:
            self.child=subprocess.Popen(command,cwd=ROOT,env=self.env,stdout=stream,stderr=subprocess.STDOUT)
            self.update(child_pid=self.child.pid)
            code=self.child.wait()
        self.child=None;entry.update(returncode=code,finished_utc=now(),status="completed" if code==0 else "failed")
        self.update(child_pid=None)
        self.check_sources()
        if code:raise RuntimeError(label+" failed; see "+str(log))

    def training(self,variant,gate):
        suffix="1e-gate" if gate else "150e"
        name=f"5090-rtdetr-r18-{variant}-{suffix}-s0"
        run=self.root/name
        cmd=[sys.executable,"-u",str(HERE/"train_fresh_budgets_5090.py"),"--variant",variant,"--project",str(self.root),
             "--name",name,"--data",str(self.root/"data.yaml"),"--plan",str(self.root/"initialization_plan.json"),
             "--reports",str(self.args.reports),"--epochs","1" if gate else "150","--fraction","0.01" if gate else "1"]
        self.phase(name,cmd,run)
        s=json.loads((run/"experiment.json").read_text())
        if s["status"]!="completed" or s["epochs_completed"]!=(1 if gate else 150) or not s["full_initial_state_verified"]:
            raise RuntimeError("Training completion/initialization gate failed")
        return run

    def evaluation(self,variant,training,gate):
        output=self.root/(training.name+"-eval")
        python=sys.executable if gate else str(self.args.deploy_python)
        cmd=[python,"-u",str(HERE/"evaluate_fresh_budgets_5090.py"),"--variant",variant,
             "--checkpoint",str(training/"weights/best.pt"),"--output",str(output)]
        cmd += ["--limit","8"] if gate else ["--benchmark"]
        self.phase(output.name,cmd,output)
        result=json.loads((output/"summary.json").read_text())
        if result["status"]!="completed" or result["protocol"]["images"]!=(8 if gate else 548) or not result["model_state_unchanged"]:
            raise RuntimeError("Evaluation completion/identity gate failed")
        return result

    def validate_gates(self):
        plan=json.loads((self.root/"initialization_plan.json").read_text())
        receipts={}
        for v in BUDGETS:
            run=self.root/f"5090-rtdetr-r18-{v}-1e-gate-s0"
            s=json.loads((run/"experiment.json").read_text())
            e=json.loads((self.root/(run.name+"-eval")/"summary.json").read_text())
            if s["status"]!="completed" or s["epochs_completed"]!=1 or s["common_state_sha256"]!=plan["common_state_sha256"] or e["protocol"]["images"]!=8 or not e["repeated_forward_exact"]:
                raise RuntimeError("Incomplete four-arm training/reload gate")
            receipts[v]={"training_state_sha256":sha256_file(run/"experiment.json"),
                         "evaluation_sha256":sha256_file(self.root/(run.name+"-eval")/"summary.json"),
                         "full_initial_state_sha256":s["initialization"]["full_initial_state_sha256"],
                         "best_checkpoint_sha256":s["checkpoints"]["best"]["sha256"]}
        if self.args.continue_after_gates and receipts!=json.loads((self.root/"gates.json").read_text())["variants"]:
            raise RuntimeError("Gate evidence changed")
        return receipts

    def run(self):
        try:
            self.prepare()
            if not self.args.continue_after_gates:
                for v in BUDGETS:
                    self.evaluation(v,self.training(v,True),True)
                receipts=self.validate_gates()
                write_atomic(self.root/"gates.json",{"status":"passed","finished_utc":now(),"variants":receipts})
                self.update(status="gates_passed",phase="all-four-gates-passed")
                if self.args.gate_only:return
            for v in BUDGETS:
                self.state["arms"][v]["status"]="running";self.update()
                try:
                    run=self.training(v,False);result=self.evaluation(v,run,False)
                    self.state["arms"][v].update(status="completed",run=str(run),metrics=result["metrics"],benchmark_status=result["benchmark"]["status"])
                except SourceChanged:raise
                except Exception as e:self.state["arms"][v].update(status="failed",error=str(e))
                self.update()
            success=all(a["status"]=="completed" and a["benchmark_status"]=="completed" for a in self.state["arms"].values())
            self.update(status="completed" if success else "completed_with_failures",phase="queue-finished",finished_utc=now())
        except BaseException as e:
            if self.child is not None and self.child.poll() is None:
                self.child.terminate()
                try:self.child.wait(timeout=15)
                except subprocess.TimeoutExpired:self.child.kill();self.child.wait()
            if self.owned:self.update(status="failed",phase="queue-failed",error=type(e).__name__+": "+str(e),child_pid=None)
            raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--reports",type=Path,default=ROOT/"reports/5090/fresh_budgets")
    p.add_argument("--deploy-python",type=Path,default=Path('/tmp/zhj-uav-profiler-20261025-venv/bin/python'))
    group=p.add_mutually_exclusive_group();group.add_argument("--gate-only",action="store_true");group.add_argument("--continue-after-gates",action="store_true")
    args=p.parse_args();args.reports=args.reports.resolve()
    if "5090" not in args.root.name:p.error("root must contain 5090")
    with Path('/tmp/zhj-uav-5090-gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import torch
        if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Queue requires the actual CUDA RTX 5090")
        Queue(args).run()
