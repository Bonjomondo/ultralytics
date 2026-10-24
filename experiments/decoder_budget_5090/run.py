"""5090 phase 0/1: explicit source pin, native observation, complete query reruns.

Raw tensors, checkpoints and predictions belong under runs/, outside Git.
No training or sampling-point surgery is performed by this entry point.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write(path, value, *, exclusive=False):
    import numpy as np
    def scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported JSON value: {type(item).__name__}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=scalar) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def source_identity(repo):
    return {"root": str(repo), "commit": git(repo, "rev-parse", "HEAD"),
            "branch": git(repo, "branch", "--show-current"),
            "dirty_paths": git(repo, "status", "--porcelain").splitlines()}


def experiment_sources():
    directory = Path(__file__).resolve().parent
    return {p.name: digest(p) for p in sorted(directory.glob("*.py"))}


def configure():
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(20261024)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def environment(device):
    import torch
    record = {"python": platform.python_version(), "executable": sys.executable,
              "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(), "device_requested": device,
              "cuda_available": torch.cuda.is_available(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_tf32": torch.backends.cudnn.allow_tf32,
              "dtype": "float32", "autocast": False}
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        record["nvidia_smi"] = {"returncode": result.returncode, "stdout": result.stdout.strip(),
                                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        record["nvidia_smi"] = {"error": str(error)}
    if device.startswith("cuda") and record["cuda_available"]:
        target = torch.device(device)
        prop = torch.cuda.get_device_properties(target)
        record.update(gpu=prop.name, capability=list(torch.cuda.get_device_capability(target)),
                      total_memory_bytes=prop.total_memory, cuda_device_index=target.index or 0)
    return record


def preflight(args):
    configure()
    record = {"schema": "decoder-budget.preflight.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source": source_identity(ROOT), "sdk": source_identity(args.sdk),
              "experiment_sources": experiment_sources(), "environment": environment(args.device), "errors": []}
    env = record["environment"]
    if args.device.startswith("cuda") and not env["cuda_available"]:
        record["errors"].append("CUDA unavailable in this process; check device access/execution isolation")
    if args.device.startswith("cuda") and args.expected_gpu not in env.get("gpu", ""):
        record["errors"].append(f"Expected GPU name containing {args.expected_gpu!r}")
    for name in ("checkpoint", "annotations", "images"):
        path = getattr(args, name, None)
        if path is not None:
            path = Path(path).resolve()
            if not (path.is_dir() if name == "images" else path.is_file()):
                record["errors"].append(f"Missing {name}: {path}")
            else:
                record[name] = {"path": str(path), **({"sha256": digest(path)} if path.is_file() else {})}
    record["status"] = "failed" if record["errors"] else "passed"
    return record


@contextmanager
def query_budget(head, budget):
    """Change native encoder Top-K, then restore even after an inference failure."""
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("Query budget must be a positive integer")
    if head.learnt_init_query:
        raise ValueError("Query sweeps require encoder-initialized content queries")
    original = head.num_queries
    try:
        head.num_queries = budget
        yield
    finally:
        head.num_queries = original


def check_budgets(budgets):
    if not budgets or any(q < 1 for q in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("Budgets must be unique positive integers")
    return sorted(budgets, reverse=True)


def summarize_coverage(images):
    """Pool GT counts, never average fractions of differently sized images."""
    groups = {}
    counts = ("denominator", "covered", "class_correct_covered", "newly_covered", "newly_class_correct_covered")
    for image in images:
        for row in image["rows"]:
            key = (row["budget"], row["iou_threshold"], row["group"])
            pooled = groups.setdefault(key, {k: row[k] for k in ("budget", "iou_threshold", "group")})
            for field in counts:
                pooled[field] = pooled.get(field, 0) + row[field]
    for row in groups.values():
        denominator = row["denominator"]
        row["coverage"] = row["covered"] / denominator if denominator else None
        row["class_correct_coverage"] = row["class_correct_covered"] / denominator if denominator else None
    return {"images": len(images), "semantics": "GT-weighted saved encoder-prefix candidate coverage; not AP or matching",
            "rows": [groups[k] for k in sorted(groups)]}


def save_coverage(root, coverage):
    import csv
    write(root / "coverage.json", coverage)
    summary = summarize_coverage(coverage)
    write(root / "coverage_summary.json", summary)
    with (root / "coverage_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["rows"][0]))
        writer.writeheader()
        writer.writerows(summary["rows"])


def input_identity(raw_dir):
    """Bind all budgets to identical preprocessed inputs, original GT and order."""
    from detrscope.core.records import iter_batches
    rows = []
    for payload in iter_batches(raw_dir):
        for image in payload["images"]:
            metadata = image["metadata"]
            targets = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in image["targets"].items()}
            rows.append({"image_id": metadata["image_id"], "input_sha256": metadata["input_sha256"],
                         "original_size": metadata["original_size"], "input_dtype": metadata["input_dtype"],
                         "normalization": metadata["normalization"], "targets": targets})
    return hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()


def tiny_model():
    import torch
    from ultralytics.nn.tasks import RTDETRDetectionModel
    cfg = {"nc": 3, "backbone": [[-1, 1, "Conv", [16, 3, 2]],
                                 [-1, 1, "Conv", [16, 3, 2]], [-1, 1, "Conv", [16, 3, 2]]],
           "head": [[[0, 1, 2], 1, "RTDETRDecoder", [3, 16, 300, 4, 4, 3, 32]]]}
    model = RTDETRDetectionModel(cfg=cfg, ch=3, nc=3, verbose=False).float().eval()
    model.names = {0: "person", 1: "car", 2: "bike"}
    # Native bbox heads start at zero; use deterministic nonzero deltas to test
    # propagation through the actual trained-head-shaped computation path.
    with torch.no_grad():
        for head in model.model[-1].dec_bbox_head:
            head.layers[-1].weight.normal_(std=0.05)
    return model


def smoke(args):
    import torch
    from detrscope import Debugger, DetectionBatch
    from detrscope.core.io import state_sha256
    from detrscope.core.records import iter_batches
    from experiments.decoder_budget_5090.source_adapter import PinnedRTDETRAdapter
    from experiments.decoder_budget_5090.coverage import coverage_for_image
    configure()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    args._created_output = root
    env = preflight(args)
    write(root / "preflight.json", env)
    if env["errors"]:
        raise RuntimeError("Preflight failed; see output/preflight.json")
    model = tiny_model().to(args.device)
    head = model.model[-1]
    names = list(model.names.values())
    adapter = PinnedRTDETRAdapter(model, repo=ROOT, pin=args.pin, class_names=names, category_ids=[1, 2, 3])
    inputs = torch.rand(2, 3, 64, 64, device=args.device)
    batch = DetectionBatch(inputs, [{"image_id": i, "original_size": [64, 64],
                                   "transformed_size": [64, 64], "normalization": {"transform": "stretch"}}
                                  for i in range(2)],
                           [{"boxes": torch.tensor([[.5, .5, .2, .2]]), "classes": torch.tensor([0]),
                             "gt_ids": [i + 1], "areas": torch.tensor([163.84])} for i in range(2)])
    before = state_sha256(model)
    results = []
    budgets = check_budgets(args.budgets)
    for budget in budgets:
        with query_budget(head, budget):
            debugger = Debugger(model, adapter, root / f"q{budget}")
            manifest = debugger.observe([batch], probes="auto", provenance={"environment": env["environment"],
                                        "experiment_sources": experiment_sources(),
                                        "scope": "random-weight tiny native model, synthetic inputs/GT; engineering validation only",
                                        "source_pin_sha256": digest(args.pin), "query_budget": budget})
            images = [im for payload in iter_batches(debugger.output_dir) for im in payload["images"]]
            if any(len(s["boxes"]) != budget for im in images for s in im["stages"]):
                raise RuntimeError("Native query budget did not reach every executed stage")
            coverage = [coverage_for_image(im, budgets=tuple(sorted(q for q in budgets if q <= budget))) for im in images]
            save_coverage(debugger.output_dir, coverage)
            debugger.analyze()
            debugger.report()
            results.append({"queries": budget, "images": manifest["image_count"],
                            "executed_layers": adapter.describe()["executed_eval_layer_ids"],
                            "identity_passed_all": all(b["identity_passed"] and b["restoration_passed"] for b in manifest["batches"]),
                            "manifest_sha256": digest(debugger.output_dir / "manifest.json")})
    if head.num_queries != 300 or state_sha256(model) != before:
        raise RuntimeError("Query sweep failed to restore model configuration/state")
    hooks = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())
    if hooks:
        raise RuntimeError("Observation hooks leaked")
    result = {"schema": "decoder-budget.smoke.v1", "status": "passed", "environment": env["environment"],
              "scope": "synthetic engineering validation; no trained checkpoint, dataset AP or speed conclusion",
              "source": source_identity(ROOT), "sdk": source_identity(args.sdk), "source_pin_sha256": digest(args.pin),
              "experiment_sources": experiment_sources(),
              "model_state_unchanged": True, "query_budget_restored": True, "hooks_remaining": hooks,
              "conditions": results}
    write(root / "summary.json", result)
    print(json.dumps({"status": result["status"], "conditions": results}))


def batches(paths, coco, device, batch_size, imgsz):
    import cv2
    import numpy as np
    import torch
    from ultralytics.data.augment import LetterBox
    from detrscope.adapters.sodetr_data import batch_from_ultralytics
    transform = LetterBox((imgsz, imgsz), auto=False, scale_fill=True)
    for start in range(0, len(paths), batch_size):
        part = paths[start:start + batch_size]
        originals = [cv2.imread(str(p)) for p in part]
        if any(im is None for im in originals):
            raise ValueError("Unreadable validation image")
        arrays = np.stack([transform(image=im) for im in originals])[..., ::-1].transpose(0, 3, 1, 2)
        inputs = torch.from_numpy(np.ascontiguousarray(arrays)).to(device).float() / 255
        yield batch_from_ultralytics({"im_file": part, "ori_shape": [im.shape[:2] for im in originals]}, inputs=inputs, coco=coco)


def export_predictions(raw_dir, model, device, imgsz, conf):
    """Run native Predictor postprocess on the verified final raw query tensors."""
    import cv2
    import torch
    from types import SimpleNamespace
    from detrscope.core.records import iter_batches
    from ultralytics.models.rtdetr.predict import RTDETRPredictor
    predictor = RTDETRPredictor(overrides={"imgsz": imgsz, "conf": conf, "max_det": 300, "verbose": False})
    predictor.model = SimpleNamespace(names=model.names)
    predictions, identities = [], []
    for payload in iter_batches(raw_dir):
        images = payload["images"]
        final = [next(s for s in reversed(im["stages"]) if s["source"] == "decoder") for im in images]
        native = torch.stack([torch.cat((s["boxes"], s["probabilities"]), -1) for s in final]).to(device)
        originals = [cv2.imread(im["metadata"]["image_path"]) for im in images]
        predictor.batch = ([im["metadata"]["image_path"] for im in images], originals, [""] * len(images))
        results = predictor.postprocess((native, None), None, originals)
        for image, result in zip(images, results):
            iid = image["metadata"]["image_id"]
            identities.append(iid)
            for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                x1, y1, x2, y2 = box
                predictions.append({"image_id": iid, "category_id": model._budget_category_ids[int(cls)],
                                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    return predictions, identities


def benchmark(model, inputs, warmup=50, repetitions=200):
    """Separate, unobserved, unfused FP32 CUDA forward; raw event samples retained."""
    import numpy as np
    import torch
    if inputs.device.type != "cuda":
        raise ValueError("Performance benchmark requires CUDA")
    if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
        raise RuntimeError("Remove observation hooks before timing")
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        torch.cuda.synchronize(inputs.device)
        torch.cuda.reset_peak_memory_stats(inputs.device)
        timings = []
        for _ in range(repetitions):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.device(inputs.device):
                start.record()
                model(inputs)
                end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
    return {"scope": "unfused FP32 batch1 GPU forward; no observer, transfer or pre/postprocess; no historical latency comparison",
            "warmup": warmup, "repetitions": repetitions, "latency_ms": timings,
            "median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(inputs.device)}


def sweep(args):
    import torch
    from detrscope import Debugger
    from detrscope.adapters.sodetr_data import CocoTargets
    from detrscope.core.io import state_sha256
    from detrscope.core.records import iter_batches
    from detrscope.evaluation.coco import CocoEvaluator, compare_gt
    from experiments.decoder_budget_5090.source_adapter import PinnedRTDETRAdapter
    from experiments.decoder_budget_5090.coverage import coverage_for_image
    configure()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    args._created_output = root
    check = preflight(args)
    write(root / "preflight.json", check)
    if check["errors"]:
        raise RuntimeError("Preflight failed; see output/preflight.json")
    # Explicit caller-supplied research checkpoint, not a downloaded model.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = (checkpoint.get("ema") or checkpoint["model"]).float().eval().to(args.device)
    names = [str(model.names[i]) for i in range(len(model.names))]
    coco = CocoTargets(args.annotations, names)
    if set(coco.categories.values()) != set(range(len(names))):
        raise ValueError("COCO categories must cover every model class exactly once")
    category_ids = [next(cid for cid, index in coco.categories.items() if index == i) for i in range(len(names))]
    model._budget_category_ids = category_ids
    adapter = PinnedRTDETRAdapter(model, repo=ROOT, pin=args.pin, class_names=names, category_ids=category_ids)
    head = adapter.head
    budgets = check_budgets(args.budgets)
    if max(budgets) != head.num_queries:
        raise ValueError("Largest budget must equal the checkpoint's native query count")
    content = json.loads(args.annotations.read_text())
    paths = []
    for image in sorted(content["images"], key=lambda im: im["id"]):
        path = args.images / image["file_name"]
        paths.append(path if path.is_file() else args.images / Path(image["file_name"]).name)
    if args.limit:
        paths = paths[:args.limit]
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError("Missing validation images")
    before = state_sha256(model)
    protocol = {"scope": "full_val" if len(paths) == len(content["images"]) else "subset_engineering_check",
                "imgsz": args.imgsz, "batch": args.batch, "dtype": "float32", "fused": False,
                "preprocess": "native LetterBox scale_fill; BGR to RGB, CHW, /255",
                "postprocess": "native RTDETRPredictor; argmax class, conf threshold, max_det=300; no JSON rounding",
                "conf": args.conf, "formal_max_dets": [1, 10, 100], "checkpoint_sha256": digest(args.checkpoint),
                "annotation_sha256": digest(args.annotations), "source_pin_sha256": digest(args.pin),
                "experiment_sources": experiment_sources(), "model_state_sha256": before,
                "environment": check["environment"], "query_budget_semantics": "native Top-K then complete Decoder rerun"}
    write(root / "protocol.json", protocol)
    evaluator = CocoEvaluator(args.annotations)
    results, baseline, baseline_inputs = [], None, None
    for budget in budgets:
        with query_budget(head, budget):
            debug = Debugger(model, adapter, root / f"q{budget}")
            manifest = debug.observe(batches(paths, coco, args.device, args.batch, args.imgsz),
                                     provenance={**protocol, "query_budget": budget})
            identity = input_identity(debug.output_dir)
            if baseline_inputs is None:
                baseline_inputs = identity
            elif identity != baseline_inputs:
                raise RuntimeError("Input/GT identity or order changed between query budgets")
            coverage = [coverage_for_image(image, budgets=tuple(sorted(q for q in budgets if q <= budget)))
                        for payload in iter_batches(debug.output_dir) for image in payload["images"]]
            save_coverage(debug.output_dir, coverage)
            predictions, image_ids = export_predictions(debug.output_dir, model, args.device, args.imgsz, args.conf)
            write(debug.output_dir / "predictions.json", predictions)
            formal = evaluator.evaluate(predictions, image_ids=image_ids)
            write(debug.output_dir / "coco.json", formal)
            if baseline is None:
                baseline = formal
            else:
                write(debug.output_dir / "formal_gt_changes.json", compare_gt(baseline, formal))
            summary = {"queries": budget, "images": manifest["image_count"], "metrics": formal["metrics"],
                       "delta_pp": {key: 100 * (value - baseline["metrics"][key]) if value >= 0 and baseline["metrics"][key] >= 0 else None
                                    for key, value in formal["metrics"].items()},
                       "manifest_sha256": digest(debug.output_dir / "manifest.json"),
                       "input_gt_identity_sha256": identity,
                       "coverage_all_gt": [row for row in summarize_coverage(coverage)["rows"]
                                           if row["budget"] == budget and row["group"] == "all"],
                       "coverage_summary_sha256": digest(debug.output_dir / "coverage_summary.json"),
                       "predictions_file_sha256": digest(debug.output_dir / "predictions.json")}
            debug.analyze()
            debug.report()
            if args.benchmark:
                x = next(batches(paths[:1], coco, args.device, 1, args.imgsz)).inputs
                timing = benchmark(model, x)
                write(debug.output_dir / "benchmark.json", timing)
                summary["benchmark_median_ms"] = timing["median_ms"]
            results.append(summary)
            write(root / "progress.json", {"status": "running", "conditions": results})
    if state_sha256(model) != before or head.num_queries != max(budgets):
        raise RuntimeError("Query sweep changed checkpoint state/configuration")
    write(root / "summary.json", {"status": "completed", "protocol": protocol, "conditions": results,
                                  "limitation": "Only same-run Q deltas are paired; historical fused/rounded 4090 results use a different protocol"})
    write(root / "progress.json", {"status": "completed", "conditions": results})
    print(json.dumps({"status": "completed", "output": str(root), "conditions": results}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, default=ROOT.parent / "detrscope")
    commands = parser.add_subparsers(dest="command", required=True)
    pin = commands.add_parser("pin")
    pin.add_argument("--expected-commit", required=True)
    pin.add_argument("--output", type=Path, required=True)
    for name in ("preflight", "smoke", "sweep"):
        command = commands.add_parser(name)
        command.add_argument("--device", default="cpu" if name == "smoke" else "cuda:0")
        command.add_argument("--expected-gpu", default="5090")
        command.add_argument("--output", type=Path, required=True)
        if name != "preflight":
            command.add_argument("--pin", type=Path, required=True)
            command.add_argument("--budgets", type=int, nargs="+", default=[300, 250, 200, 150])
        if name != "smoke":
            for flag in ("checkpoint", "annotations", "images"):
                command.add_argument("--" + flag, type=Path, required=name == "sweep")
        if name == "sweep":
            command.add_argument("--limit", type=int, default=0, help="0 means full validation set")
            command.add_argument("--batch", type=int, default=2)
            command.add_argument("--imgsz", type=int, default=640)
            command.add_argument("--conf", type=float, default=0.001)
            command.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs/decoder_budget_5090/ultralytics-config"))
    args.sdk = args.sdk.resolve()
    sys.path.insert(0, str(args.sdk / "src"))
    import detrscope
    if not Path(detrscope.__file__).resolve().is_relative_to(args.sdk / "src"):
        raise RuntimeError("Imported detrscope from the wrong source tree")
    if args.command == "pin":
        from experiments.decoder_budget_5090.source_adapter import create_source_pin
        create_source_pin(ROOT, args.output, expected_commit=args.expected_commit)
        print(json.dumps({"pin": str(args.output), "sha256": digest(args.output)}))
    elif args.command == "preflight":
        if args.output.exists():
            raise FileExistsError(args.output)
        result = preflight(args)
        write(args.output, result, exclusive=True)
        print(json.dumps({"status": result["status"], "errors": result["errors"]}))
        if result["errors"]:
            raise SystemExit(1)
    else:
        if args.command == "sweep" and (args.limit < 0 or args.batch < 1 or args.imgsz < 32 or not 0 <= args.conf <= 1):
            parser.error("Require limit>=0, batch>=1, imgsz>=32 and conf in [0,1]")
        try:
            (smoke if args.command == "smoke" else sweep)(args)
        except Exception as error:
            if (getattr(args, "_created_output", None) == args.output.resolve()
                    and not (args.output / "summary.json").exists()):
                failure = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                write(args.output / "failure.json", failure)
                progress = args.output / "progress.json"
                if progress.exists():
                    write(progress, {**json.loads(progress.read_text()), **failure})
            raise


if __name__ == "__main__":
    main()
