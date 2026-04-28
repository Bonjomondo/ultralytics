# VisDrone + YOLO Experiment Guide

## 1. Drone detection datasets

If your target is generic drone-view object detection, `VisDrone` is still the first dataset to run.

- `VisDrone`: the standard baseline for drone-view detection in YOLO papers, especially for small objects.
- `DOTA`: use it when you need oriented bounding boxes or remote-sensing style scenes.
- `UAVDT`: use it when your paper leans toward tracking or traffic scenes.

For a first detection paper or reproduction run in this repo, start with `VisDrone`.

## 2. Prepare the dataset in this repo

This repo now includes a workspace-local `VisDrone-local.yaml`, so you can keep the dataset inside the current repo.

From the repo root:

```powershell
python tools/prepare_visdrone.py
```

Default output:

```text
datasets/VisDrone/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

If you want a custom location:

```powershell
python tools/prepare_visdrone.py --root D:\datasets\VisDrone
```

## 3. Install the training environment

This workspace currently does not have the minimum runtime packages for training.

Recommended:

```powershell
pip install -e .
```

At minimum, you need `torch`, `torchvision`, `opencv-python`, and `pillow`.

## 4. Recommended experiment order

Do not start by stacking random attention blocks. On VisDrone, the gain usually comes from better small-object coverage first.

Recommended order:

1. Run a clean baseline with a pretrained official model.
2. Add a `P2` detection head for smaller targets.
3. Raise `imgsz` to `960` or `1280` if GPU memory allows.
4. Tune augmentation and training schedule.
5. Only then test new attention or fusion modules.

## 5. Strong baseline commands

Baseline:

```powershell
yolo detect train data=VisDrone-local.yaml model=yolo11n.pt imgsz=960 epochs=150 batch=16 device=0 workers=8 project=runs/visdrone name=yolo11n_960 seed=42
```

Small-object baseline with `P2`:

```powershell
yolo detect train data=VisDrone-local.yaml model=ultralytics/cfg/models/11/yolo11-p2.yaml imgsz=960 epochs=150 batch=16 device=0 workers=8 project=runs/visdrone name=yolo11n_p2_960 seed=42
```

Validation:

```powershell
yolo detect val data=VisDrone-local.yaml model=runs/visdrone/yolo11n_p2_960/weights/best.pt split=val imgsz=960 device=0
```

Inference visualization:

```powershell
yolo detect predict model=runs/visdrone/yolo11n_p2_960/weights/best.pt source=datasets/VisDrone/images/val imgsz=960 conf=0.25 save=True project=runs/visdrone name=pred_val
```

## 6. What to change first for VisDrone

For VisDrone, the most defensible modifications are:

- Add `P2` output.
- Increase input size.
- Reduce overly aggressive downsampling.
- Improve neck feature fusion.
- Then try attention modules.

If your paper claims small-object improvement, `P2` is usually a more serious baseline than adding attention alone.

## 7. Use existing modules already in the repo

Many modules are already implemented under `ultralytics/nn/modules/`, so you do not need to write new code for every ablation.

Examples:

- `C2PSA` in `ultralytics/nn/modules/block.py`
- `CBAM` in `ultralytics/nn/modules/conv.py`
- `C3k2` in `ultralytics/nn/modules/block.py`

In this workspace, `CBAM` has now been wired into YAML model parsing, so you can insert it directly in a model YAML.

Example snippet:

```yaml
- [-1, 1, C3k2, [256, False]]
- [-1, 1, CBAM, [7]]
```

That keeps channel count unchanged and adds attention after the block.

## 8. How YAML-based model changes work here

The key parser is `ultralytics/nn/tasks.py` in `parse_model()`.

- `ultralytics/cfg/models/.../*.yaml`: defines the network structure.
- `ultralytics/nn/modules/*.py`: stores module implementations.
- `ultralytics/nn/modules/__init__.py`: exports modules.
- `ultralytics/nn/tasks.py`: imports modules and maps YAML names to Python classes.

If the module already exists and is registered, you only need to edit the YAML.

## 9. How to add a brand-new module

If the module does not exist yet, the shortest safe path is:

1. Implement the module in `ultralytics/nn/modules/block.py` or `ultralytics/nn/modules/conv.py`.
2. Export it in `ultralytics/nn/modules/__init__.py`.
3. Import it in `ultralytics/nn/tasks.py`.
4. Register its argument pattern in `parse_model()`.
5. Add it to your model YAML and train.

The parser step matters:

- If the module changes channels, add handling like other `base_modules`.
- If the module keeps input and output channels the same, a special case like `CBAM` is often enough.
- If the module has an internal repeat count, mirror the handling used by `repeat_modules`.

## 10. A good ablation table for this repo

Do not compare ten changes at once. Use a sequence like this:

| ID | Change | imgsz | mAP50-95 | Params | FLOPs |
|---|---|---:|---:|---:|---:|
| A0 | YOLO11n baseline | 960 |  |  |  |
| A1 | A0 + P2 head | 960 |  |  |  |
| A2 | A1 + CBAM | 960 |  |  |  |
| A3 | A2 + tuned augmentation | 960 |  |  |  |
| A4 | A3 + imgsz 1280 | 1280 |  |  |  |

This makes the contribution chain readable.

## 11. Practical advice

- Keep the same train/val/test split for all experiments.
- Fix seeds like `42`, `43`, `44` and report the mean.
- Record `imgsz`, `batch`, `epochs`, and `device` in every run name.
- Save both `best.pt` and the full `results.csv`.

## 12. What to do next

The next sensible steps in this repo are:

1. Install the runtime dependencies.
2. Run `python tools/prepare_visdrone.py`.
3. Train `yolo11n.pt`.
4. Train `yolo11-p2.yaml`.
5. If you want, add `CBAM` in the neck and run the ablation.
