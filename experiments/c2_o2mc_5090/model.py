"""Training-only PP-YOLOE-style O2M-C for the unchanged local C2 RT-DETR.

PyTorch adaptation informed by clxia12/RT-DETRv3 (Apache-2.0),
ppdet/modeling/{heads/ppyoloe_head.py,assigners/*}. No Paddle dependency.
This is an O2M-C experiment, not a reproduction of the full RT-DETRv3.
"""
from copy import copy, deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.r18_p2_sfi.train import MODELS, build_model, tensor_digest
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import Conv
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.metrics import bbox_iou, box_iou
from ultralytics.utils.ops import xywh2xyxy


class ESEStem(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, 1)
        self.conv = Conv(channels, channels, 1)
        nn.init.normal_(self.fc.weight, std=.001)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.conv(x * self.fc(F.adaptive_avg_pool2d(x, 1)).sigmoid())


class DenseHead(nn.Module):
    """Three unshared ESE cls/reg stems; 17 bins for each LTRB distance."""
    def __init__(self, nc=10, channels=256, reg_max=16):
        super().__init__()
        self.nc, self.reg_max = nc, reg_max
        self.cls_stems = nn.ModuleList(ESEStem(channels) for _ in range(3))
        self.reg_stems = nn.ModuleList(ESEStem(channels) for _ in range(3))
        self.cls_preds = nn.ModuleList(nn.Conv2d(channels, nc, 3, padding=1) for _ in range(3))
        self.reg_preds = nn.ModuleList(nn.Conv2d(channels, 4 * (reg_max + 1), 3, padding=1) for _ in range(3))
        self.register_buffer("projection", torch.arange(reg_max + 1, dtype=torch.float32))
        for cls, reg in zip(self.cls_preds, self.reg_preds):
            nn.init.zeros_(cls.weight)
            nn.init.constant_(cls.bias, math.log(.01 / .99))
            nn.init.zeros_(reg.weight)
            nn.init.ones_(reg.bias)

    def forward(self, features):
        if len(features) != 3:
            raise ValueError("O2M-C requires the native P3/P4/P5 decoder inputs")
        logits, distributions, points, strides, counts = [], [], [], [], []
        for i, (feature, stride) in enumerate(zip(features, (8, 16, 32))):
            logits.append(self.cls_preds[i](self.cls_stems[i](feature) + feature).flatten(2).transpose(1, 2))
            distributions.append(self.reg_preds[i](self.reg_stems[i](feature)).flatten(2).transpose(1, 2))
            h, w = feature.shape[-2:]
            yy, xx = torch.meshgrid(torch.arange(h, device=feature.device, dtype=feature.dtype) + .5,
                                    torch.arange(w, device=feature.device, dtype=feature.dtype) + .5, indexing="ij")
            points.append(torch.stack((xx, yy), -1).reshape(-1, 2))
            strides.append(feature.new_full((h * w, 1), stride))
            counts.append(h * w)
        logits, distributions = torch.cat(logits, 1), torch.cat(distributions, 1)
        points, strides = torch.cat(points), torch.cat(strides)
        distances = distributions.reshape(*logits.shape[:2], 4, self.reg_max + 1).softmax(-1) @ self.projection
        boxes = torch.cat((points - distances[..., :2], points + distances[..., 2:]), -1)
        return logits, distributions, boxes, points, strides, counts


@torch.no_grad()
def assign(scores, boxes, points, strides, counts, labels, gt, static):
    """One image at a time to bound dense-scene assignment memory.

ATSS: top9 nearest per level, mean + unbiased std of anchor IoU,
5*stride square anchors. TAL: top13, score^1 * IoU^6. Conflicts
follow the reference's highest-IoU GT rule; no forced GT matching.
"""
    n, nc = scores.shape
    target_scores, target_boxes = scores.new_zeros((n, nc)), boxes.new_zeros((n, 4))
    assigned = torch.full((n,), -1, dtype=torch.long, device=boxes.device)
    if not len(gt):
        return target_scores, target_boxes, assigned
    pixel_points = points * strides
    inside = torch.cat((pixel_points[None] - gt[:, None, :2],
                        gt[:, None, 2:] - pixel_points[None]), -1).amin(-1) > 1e-9
    predicted_iou = box_iou(gt, boxes * strides)
    if static:
        anchors = torch.cat((pixel_points - 2.5 * strides, pixel_points + 2.5 * strides), -1)
        overlaps = box_iou(gt, anchors)
        centers = (gt[:, :2] + gt[:, 2:]) / 2
        distances = (centers[:, None] - pixel_points[None]).square().sum(-1)
        candidates, offset = [], 0
        for count in counts:
            candidates.append(distances[:, offset:offset + count].topk(min(9, count), largest=False).indices + offset)
            offset += count
        indices = torch.cat(candidates, -1)
        candidate_iou = overlaps.gather(1, indices)
        threshold = candidate_iou.mean(-1, keepdim=True) + candidate_iou.std(-1, keepdim=True)
        positive = torch.zeros_like(inside).scatter_(1, indices, True)
        positive &= inside & (overlaps > threshold)
    else:
        overlaps = predicted_iou
        alignment = scores[:, labels].T * overlaps.pow(6)
        indices = (alignment * inside).topk(min(13, n), dim=-1).indices
        positive = torch.zeros_like(inside).scatter_(1, indices, True) & inside
    conflicts = positive.sum(0) > 1
    winners = F.one_hot(overlaps.argmax(0), len(gt)).T.bool()
    positive = torch.where(conflicts[None], winners, positive)
    foreground = positive.any(0)
    assigned[foreground] = positive[:, foreground].float().argmax(0)
    target_boxes[foreground] = gt[assigned[foreground]]
    if static:
        quality = (predicted_iou * positive).amax(0)
    else:
        alignment *= positive
        quality = (alignment / (alignment.amax(-1, keepdim=True) + 1e-9)
                   * (overlaps * positive).amax(-1, keepdim=True)).amax(0)
    rows = foreground.nonzero(as_tuple=True)[0]
    target_scores[rows, labels[assigned[rows]]] = quality[rows]
    return target_scores, target_boxes, assigned


def dense_loss(outputs, batch, epoch, switch_epoch=30, reg_max=16, diagnostics=False):
    logits, distributions, boxes, points, strides, counts = outputs
    bs, n, nc = logits.shape
    h, w = batch["img"].shape[-2:]
    gt_boxes = xywh2xyxy(batch["bboxes"].to(logits.device)) * boxes.new_tensor([w, h, w, h])
    batch_indices = batch["batch_idx"].view(-1).to(logits.device)
    classes = batch["cls"].view(-1).to(device=logits.device, dtype=torch.long)
    target_scores, target_boxes, assignments, gt_counts, gt_sizes = [], [], [], [], []
    for i in range(bs):
        selected = batch_indices == i
        labels = classes[selected]
        gt = gt_boxes[selected]
        scores, bboxes, assigned = assign(logits[i].detach().sigmoid(), boxes[i].detach(), points,
                                          strides, counts, labels, gt, epoch < switch_epoch)
        target_scores.append(scores)
        target_boxes.append(bboxes / strides)
        assignments.append(assigned)
        gt_counts.append(len(gt))
        gt_sizes.append((gt[:, 2:] - gt[:, :2]).prod(-1))
    target_scores, target_boxes = torch.stack(target_scores), torch.stack(target_boxes)
    assignments = torch.stack(assignments)
    foreground = assignments >= 0
    # Explicit assigned labels, including zero-quality foreground, for VFL.
    onehot = torch.zeros_like(logits)
    for i in range(bs):
        labels = classes[batch_indices == i]
        rows = foreground[i].nonzero(as_tuple=True)[0]
        onehot[i, rows, labels[assignments[i, rows]]] = 1
    normalizer = target_scores.sum().clamp_min(1)
    weight = .75 * logits.sigmoid().detach().pow(2) * (1 - onehot) + target_scores * onehot
    vfl = (F.binary_cross_entropy_with_logits(logits, target_scores, reduction="none") * weight).sum() / normalizer
    giou, dfl = boxes.sum() * 0, distributions.sum() * 0
    if foreground.any():
        quality = target_scores.sum(-1)[foreground]
        giou = ((1 - bbox_iou(boxes[foreground], target_boxes[foreground], xywh=False, GIoU=True).view(-1))
                * quality).sum() / normalizer
        target_dist = torch.cat((points[None] - target_boxes[..., :2],
                                 target_boxes[..., 2:] - points[None]), -1)[foreground].clamp(0, reg_max - .01)
        left = target_dist.long()
        frac = target_dist - left
        pred = distributions[foreground].reshape(-1, reg_max + 1)
        ce_left = F.cross_entropy(pred, left.flatten(), reduction="none").view(-1, 4)
        ce_right = F.cross_entropy(pred, (left + 1).flatten(), reduction="none").view(-1, 4)
        dfl = (((1 - frac) * ce_left + frac * ce_right).mean(-1) * quality).sum() / normalizer
    stats = {}
    if diagnostics:
        per_gt = torch.cat([torch.bincount(a[a >= 0], minlength=g) for a, g in zip(assignments, gt_counts)])
        sizes = torch.cat(gt_sizes)
        stats = {"assigner": "ATSS" if epoch < switch_epoch else "TAL", "gt": sum(gt_counts),
                 "positives": int(foreground.sum()), "unassigned_gt": int((per_gt == 0).sum()),
                 "positives_per_gt": float(per_gt.float().mean()) if len(per_gt) else 0.,
                 "target_score_sum": float(target_scores.sum()),
                 "positives_by_level": [int(v.sum()) for v in foreground.split(counts, dim=1)],
                 "gt_by_resized_area": {}, "positives_by_resized_area": {}}
        for name, mask in (("small", sizes < 32**2), ("medium", (sizes >= 32**2) & (sizes < 96**2)),
                           ("large", sizes >= 96**2)):
            stats["gt_by_resized_area"][name] = int(mask.sum())
            stats["positives_by_resized_area"][name] = int(per_gt[mask].sum())
    return (vfl, giou, dfl), stats


class O2MCModel(RTDETRDetectionModel):
    """Inference inherits the native model unchanged; only loss captures features."""
    def __init__(self, cfg, nc=10, ch=3, verbose=False):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        self.nc = nc
        self.aux_head = DenseHead(nc)
        self.aux_weight, self.switch_epoch, self.train_epoch = .5, 30, 0
        self.capture_diagnostics = False
        self.latest_diagnostics = None

    def loss(self, batch, preds=None):
        if not self.training or self.aux_weight == 0:
            loss, items = super().loss(batch, preds)
            return loss, torch.cat((items, items.new_zeros(3)))
        if preds is not None:
            raise ValueError("Training O2M-C needs the native forward features; do not pass cached predictions")
        captured = {}

        def collect(module, args):
            captured["features"] = args[0]
            captured["outputs"] = self.aux_head(args[0])

        handle = self.model[-1].register_forward_pre_hook(collect)
        try:
            main, items = super().loss(batch)
        finally:
            handle.remove()
        components, stats = dense_loss(captured["outputs"], batch, self.train_epoch, self.switch_epoch,
                                        diagnostics=self.capture_diagnostics)
        vfl, giou, dfl = components
        auxiliary = vfl + 2.5 * giou + .5 * dfl
        if self.capture_diagnostics:
            stats.update(main_loss=float(main.detach()), auxiliary_loss=float(auxiliary.detach()),
                         weighted_aux_main_ratio=float((self.aux_weight * auxiliary / main.clamp_min(1e-9)).detach()))
            probes = [self.model[14].conv.weight, self.model[16].cv1.conv.weight, captured["features"][0]]
            gm = torch.autograd.grad(main, probes, retain_graph=True)
            ga = torch.autograd.grad(self.aux_weight * auxiliary, probes, retain_graph=True)
            stats["gradients"] = {}
            for name, a, b in zip(("p2_downsample", "p3_repc3", "fused_p3"), gm, ga):
                a, b = a.detach().float().flatten(), b.detach().float().flatten()
                na, nb = a.norm(), b.norm()
                stats["gradients"][name] = {"main_norm": float(na), "weighted_aux_norm": float(nb),
                    "cosine": float(a.dot(b) / (na * nb)) if na > 0 and nb > 0 else None}
            self.latest_diagnostics = stats
        return main + self.aux_weight * auxiliary, torch.cat((items, torch.stack([v.detach() for v in components])))


def build_o2mc(cfg, nc=10, ch=3, seed=0, verbose=False):
    reference = build_model(cfg, nc=nc, ch=ch, seed=seed, verbose=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 4000)
        model = O2MCModel(cfg, nc=nc, ch=ch, verbose=verbose)
    model.model.load_state_dict(reference.model.state_dict(), strict=True)
    model.ablation_initialization = {**reference.ablation_initialization, "aux_seed": seed + 4000,
        "c2_full_state_sha256": tensor_digest(reference.state_dict()),
        "aux_state_sha256": tensor_digest(model.aux_head.state_dict()),
        "native_c2_state_verified": tensor_digest(model.model.state_dict()) == tensor_digest(reference.model.state_dict())}
    return model


class O2MCTrainer(RTDETRTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is not None:
            raise ValueError("From-scratch entry only; interrupted runs need an explicit recovery audit")
        return build_o2mc(cfg, self.data["nc"], self.data["channels"], self.args.seed, verbose)

    def get_validator(self):
        from ultralytics.models.rtdetr.val import RTDETRValidator
        self.loss_names = "giou_loss", "cls_loss", "l1_loss", "aux_vfl", "aux_giou", "aux_dfl"
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def final_eval(self):
        import shutil
        if self.last.exists():
            shutil.copy2(self.last, self.wdir / "resume.pt")
        super().final_eval()


def strip_auxiliary(model):
    """Produce a native RTDETRDetectionModel for deployment, with strict key loading."""
    native = RTDETRDetectionModel(model.yaml, nc=model.nc, verbose=False)
    native.model.load_state_dict(model.model.state_dict(), strict=True)
    # Cached decoder anchors/shapes are not registered state_dict buffers.
    # Native checkpoints round these anchors through FP16. Preserve the entire
    # main module tree so export is identical to that reloaded checkpoint.
    native.model = deepcopy(model.model)
    native.names = model.names
    native.nc = model.nc
    native.args = model.args
    return native.cpu().float().eval()
