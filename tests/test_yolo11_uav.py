# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import (
    FrequencyEnhancedFusion,
    FrequencyFocusedDownsample,
    LightweightAlignedConcat,
    UAVDetect,
)
from ultralytics.utils.loss import SmallObjectBboxLoss, UAVDetectionLoss


def test_inner_siou_similarity():
    """Inner-SIoU should be one for identical boxes and decrease after a spatial perturbation."""
    criterion = SmallObjectBboxLoss()
    target = torch.tensor([[1.0, 1.0, 5.0, 5.0], [2.0, 3.0, 8.0, 9.0]])
    identical = criterion._inner_siou(target, target)
    shifted = criterion._inner_siou(target + torch.tensor([1.0, 0.0, 1.0, 0.0]), target)

    assert torch.allclose(identical, torch.ones_like(identical), atol=1e-6)
    assert torch.isfinite(shifted).all()
    assert (shifted < identical).all()


def test_uav_frequency_and_alignment_modules():
    """FFD, frequency fusion, and aligned concat should preserve their documented shapes and gradients."""
    downsample = FrequencyFocusedDownsample(32, 64)
    fusion = FrequencyEnhancedFusion(64, 64)
    alignment = LightweightAlignedConcat((32, 32))
    x = torch.randn(2, 32, 32, 32, requires_grad=True)

    downsampled = downsample(x)
    assert downsampled.shape == (2, 64, 16, 16)
    enhanced = fusion(downsampled)
    assert enhanced.shape == downsampled.shape
    aligned = alignment((x, torch.randn_like(x)))
    assert aligned.shape == (2, 64, 32, 32)

    (enhanced.mean() + aligned.mean()).backward()
    assert x.grad is not None
    assert downsample.frequency_scale.grad is not None
    assert fusion.scale.grad is not None
    assert alignment.align_scale.grad is not None


def test_yolo11s_uav_build_forward_and_loss():
    """The UAV YAML should expose P2-P5 outputs and select the small-object regression criterion."""
    model = YOLO("ultralytics/cfg/models/11/yolo11s-uav.yaml").model
    assert isinstance(model.model[-1], UAVDetect)
    assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]

    model.args = get_cfg()
    model.train()
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.tensor([0, 0, 1, 1]),
        "cls": torch.tensor([[0.0], [1.0], [0.0], [2.0]]),
        "bboxes": torch.tensor(
            [
                [0.25, 0.25, 0.08, 0.08],
                [0.70, 0.60, 0.25, 0.20],
                [0.40, 0.45, 0.06, 0.05],
                [0.75, 0.75, 0.18, 0.16],
            ]
        ),
    }
    predictions = model(batch["img"])
    assert [feature.shape[-2:] for feature in predictions["feats"]] == [(16, 16), (8, 8), (4, 4), (2, 2)]

    criterion = model.init_criterion()
    assert isinstance(criterion, UAVDetectionLoss)
    assert isinstance(criterion.bbox_loss, SmallObjectBboxLoss)
    loss, loss_items = model.loss(batch, predictions)
    assert torch.isfinite(loss).all()
    assert torch.isfinite(loss_items).all()
    loss.sum().backward()
