# Frozen author engine

SO-DETR author upstream: https://github.com/ValiantDiligent/SO_DETR
Upstream commit: 148d5972bbaec298b3975ca00c00d91abbc6657c.
Ultralytics 8.0.201, AGPL-3.0. This is the existing B0/B1 reconstructed-baseline engine,
copied from rtdetr-r18-c2-author-recipe-20261028. Ordinary IoU/GIoU restored;
unrelated broken YOLOv10 loader alias removed. New changes: p2_channel_gate.py
and parser import/Concat channel accounting in nn/tasks.py only.
Original backend file hashes are in ../author_backend_original_manifest.json.
All backend source and model/data configs are tracked by Git, not just server-only snapshots.
