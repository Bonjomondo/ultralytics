# R2 NAIFI screening — 2026-10-21

Server: zjnu-3660 (RTX 4090), repository branch `rtdetr-r18-na-ablation`, existing UAV_CV Python 3.11 / torch 2.1.2+cu121. Uses the existing masked F.unfold attention implementation; this phase compares accuracy, without claiming inference speed gains.

| Order | Variant | P5 replacement | Relative position bias |
| --- | --- | --- | --- |
| 1 | R2a | NAIFI, 7×7 | off |
| 2 | R2b (main candidate) | NAIFI, 7×7 | on |
| 3 | R2c | NAIFI, 11×11 | on |

All remaining model layers match R0, including Identity at layer 23 and the 3-layer decoder at layer 24. NAIFI uses post-residual LayerNorm2d and a 256→1024→256 GELU FFN. No additional gate. All non-RPB NAIFI initial tensors are identical across the three variants. The inherited R0 AIFI uses its original positional encoding; R2 uses the specified RPB variant. This is an encoder adaptation comparison, not an exact reproduction of Le-DETR.

Protocol: full VisDrone train split, 150 epochs, batch=8, imgsz=640, AdamW, lr0=lrf-adjusted constant LR=0.0001, weight_decay=0.0001, momentum=0.9, seed=0, FP32/amp=False, workers=0, nbs=64, warmup_epochs=3, warmup_bias_lr=0, mixup=0.2, no early stopping, from scratch. Other settings inherit the same runner used for completed R0/R1. Existing R0/R1 runs are reference controls; no extra baseline training is scheduled.

Initialization copies only same-name/same-shape states from a seeded R0, asserts equality, and records both selected shared-state digests and full R0 reference digest. The full reference digest was matched to the completed R0 initialization. Removed/new states are confined to model.6. Resume checks exact backbone/head architecture, including kernel and RPB.

Only model construction, shared initialization and one small GPU module backward were checked before launching full training. No added test suite or separate smoke training.

Queue runs serially on GPU 0 under tmux `r2-naifi-20261021`. Each completed best.pt is evaluated through the same native fused FP32 Validator JSON path as the completed R0/R1 comparison, batch=2, full 548-image validation split, original COCO annotations and pycocotools maxDets=[1,10,100]. Native training mAP is retained separately from formal COCO AP. Evaluator asserts the recorded baseline protocol matches. The queue stops on errors and refuses to overwrite runs. It does not restart after a server reboot.

Artifacts in this directory:

- `queue_status.json`: active phase, PID, log, completed phases or failure reason.
- `source_before.tar.gz`: original files before implementation; `base_commit.txt`: repository starting commit.
- `source_snapshot.tar.gz`, `source_sha256.json`: code and model configuration for this queue.
- `initialization_preflight.json`: three model builds and parameter identity records.
- `r2*_formal_metrics.json`, `formal_metrics_comparison.csv`: generated after each full training and evaluation.

Training outputs: `runs/detect/r18_na_ablation/rtdetr-r18-r2{a,b,c}-150e-s0-20261021/`.

Logs: `logs/rtdetr-r18-r2{a,b,c}-150e-s0-20261021-{training,evaluation}.log`.

Connect to queue: `tmux attach -t r2-naifi-20261021` (use `/home/zjnu/anaconda3/bin/tmux` if tmux is not on PATH). Training writes to log files, so use `tail -f` on the active log for batch progress.

A 300-epoch/multiple-seed follow-up is not queued; choose it after screening results.

## 2026-10-22 evaluation serialization repair and continuation

Converted COCO image/category IDs to built-in integers and native metric values to floats before JSON serialization. Added `run_queue.py --resume`: verifies the original training-source manifest, validates and skips completed phases, preserves existing runs/logs, and uses a process lock to prevent duplicate queues. R2a training and checkpoints are preserved; its evaluation is rerun, followed by the original R2b/R2c training and evaluation sequence. Prior scripts and failed state are retained under `repair_backups/`; checkpoint and repair hashes are recorded in `repair_20261022.json`.
