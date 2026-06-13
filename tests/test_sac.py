# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules import SACDetect, SemanticAlignmentCalibration


def test_semantic_alignment_calibration_shape_and_gradient():
    """SAC should preserve the spatial feature shape and backpropagate through both inputs."""
    module = SemanticAlignmentCalibration((32, 64))
    spatial = torch.randn(2, 32, 16, 16, requires_grad=True)
    semantic = torch.randn(2, 64, 4, 4, requires_grad=True)

    output = module((spatial, semantic))
    assert output.shape == spatial.shape
    output.mean().backward()
    assert spatial.grad is not None
    assert semantic.grad is not None


def test_yolo11s_sac_build_and_forward():
    """The YOLO11s-SAC YAML should build at s scale and produce training predictions."""
    model = YOLO("ultralytics/cfg/models/11/yolo11s-sac.yaml").model
    assert isinstance(model.model[-1], SACDetect)
    assert model.model[-1].sac_scale.item() == 0.0

    model.train()
    output = model(torch.randn(1, 3, 64, 64))
    assert output["boxes"].shape[0] == 1
    assert output["scores"].shape[0] == 1


def test_sac_parameters_are_compatible_with_musgd():
    """MuSGD should use Muon only for supported 2D and 4D parameters."""
    model = YOLO("ultralytics/cfg/models/11/yolo11s-sac.yaml").model
    trainer = object.__new__(BaseTrainer)
    trainer.data = {"nc": 10}
    trainer.args = type("Args", (), {"lr0": 0.01, "momentum": 0.937, "warmup_bias_lr": 0.1})()

    optimizer = trainer.build_optimizer(model, name="MuSGD", lr=0.01, momentum=0.9)
    muon_parameters = [
        parameter
        for group in optimizer.param_groups
        if group["use_muon"]
        for parameter in group["params"]
    ]
    assert muon_parameters
    assert all(parameter.ndim in {2, 4} for parameter in muon_parameters)
    muon_parameter_ids = {id(parameter) for parameter in muon_parameters}
    assert id(model.model[-1].sac.frequency_enhancer.alpha) not in muon_parameter_ids
    assert id(model.model[-1].sac.frequency_enhancer.beta) not in muon_parameter_ids
