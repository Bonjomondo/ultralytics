"""Assignment, gradient, inference, serialization and native deployment gates."""
import io

import torch

from experiments.c2_o2mc_5090.model import DenseHead, assign, build_o2mc, dense_loss, strip_auxiliary
from experiments.r18_p2_sfi.train import MODELS, build_model, tensor_digest

torch.set_num_threads(4)


def batch():
    return {"img": torch.rand(2, 3, 128, 128), "batch_idx": torch.tensor([0., 0., 1.]),
            "cls": torch.tensor([[0.], [2.], [1.]]),
            "bboxes": torch.tensor([[.4, .4, .25, .25], [.7, .7, .15, .2], [.5, .5, .4, .3]])}


def test_dense_assignment_and_losses():
    head = DenseHead(nc=3)
    inputs = [torch.randn(2, 256, s, s, requires_grad=True) for s in (16, 8, 4)]
    data = batch()
    for epoch in (0, 30):
        parts, stats = dense_loss(head(inputs), data, epoch, diagnostics=True)
        assert all(torch.isfinite(part) and part >= 0 for part in parts)
        assert stats["positives"] > 0
        assert sum(stats["positives_by_level"]) == stats["positives"]
        assert sum(stats["positives_by_resized_area"].values()) == stats["positives"]
        assert stats["gt"] == 3
    empty = {**data, "batch_idx": torch.empty(0), "cls": torch.empty(0, 1), "bboxes": torch.empty(0, 4)}
    parts, stats = dense_loss(head(inputs), empty, 0, diagnostics=True)
    assert parts[0] > 0 and parts[1] == parts[2] == 0 and stats["positives"] == 0
    sum(parts).backward()
    assert head.cls_preds[0].weight.grad.abs().sum() > 0


def test_known_one_to_many_assignment():
    # Perfect dense predictions with a single GT: TAL assigns multiple positions,
    # soft quality is one, and no gradient flows through the target assignment.
    yy, xx = torch.meshgrid(torch.arange(5) + .5, torch.arange(5) + .5, indexing="ij")
    points = torch.stack((xx, yy), -1).reshape(-1, 2)
    gt = torch.tensor([[0., 0., 5., 5.]])
    scores, boxes, ids = assign(torch.full((25, 2), .9), gt.repeat(25, 1), points,
                                torch.ones(25, 1), [25], torch.tensor([1]), gt, False)
    assert (ids >= 0).sum() == 13
    assert torch.allclose(scores[:, 1][ids >= 0], torch.ones(13))
    assert not scores.requires_grad and not boxes.requires_grad
    assert scores[:, 0].sum() == 0
    # Conflicting identical GTs must not duplicate the same positive location.
    scores, boxes, ids = assign(torch.full((25, 2), .9), gt.repeat(25, 1), points,
                                torch.ones(25, 1), [25], torch.tensor([0, 1]), gt.repeat(2, 1), False)
    assert (scores > 0).sum(-1).max() <= 1


def test_main_identity_training_gradients_and_deployment():
    cfg = str(MODELS["c0"])
    model = build_o2mc(cfg)
    plain = build_model(cfg)
    plain.nc = 10
    model.names, model.args = plain.names, {}
    assert model.ablation_initialization["native_c2_state_verified"]
    assert tensor_digest(model.model.state_dict()) == tensor_digest(plain.model.state_dict())
    data = batch()
    model.eval()
    plain.eval()

    def forbidden(*args):
        raise AssertionError("auxiliary head executed during inference")

    handle = model.aux_head.register_forward_pre_hook(forbidden)
    with torch.no_grad():
        torch.testing.assert_close(model(data["img"])[0], plain(data["img"])[0], rtol=0, atol=0)
        model.model[-1].export = True
        assert model(data["img"]).shape == (2, 300, 14)
        model.model[-1].export = False
        loss, items = model(data)
        assert torch.isfinite(loss) and items.shape == (6,) and items[-3:].sum() == 0
    handle.remove()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # Official zero predictor weights imply zero feature gradient on step one.
    # Step two must demonstrate a real auxiliary gradient into the unchanged C2.
    loss, _ = model(data)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    model.capture_diagnostics = True
    loss, items = model(data)
    assert torch.isfinite(loss) and items.shape == (6,)
    assert not model.model[-1]._forward_pre_hooks
    record = model.latest_diagnostics
    for gradient in record["gradients"].values():
        assert gradient["weighted_aux_norm"] > 0
        assert gradient["main_norm"] > 0
        assert -1.00001 <= gradient["cosine"] <= 1.00001
    loss.backward()
    assert model.model[14].conv.weight.grad.abs().sum() > 0
    assert model.aux_head.cls_preds[0].weight.grad.abs().sum() > 0
    optimizer.zero_grad()
    model.train_epoch = 30
    loss, _ = model(data)
    loss.backward()
    assert torch.isfinite(loss) and model.latest_diagnostics["assigner"] == "TAL"
    model.eval()
    # Simulate the native trainer's FP16 serialization of unregistered anchors.
    # Rebuilding anchors in FP32 would fail this deployment identity gate.
    model.half().float()
    native = strip_auxiliary(model)
    assert not any("aux" in k for k in native.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(model(data["img"])[0], native(data["img"])[0], rtol=0, atol=0)
    stream = io.BytesIO()
    torch.save(model, stream)
    stream.seek(0)
    loaded = torch.load(stream, weights_only=False)
    assert tensor_digest(loaded.state_dict()) == tensor_digest(model.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(loaded(data["img"])[0], native(data["img"])[0], rtol=0, atol=0)


def test_zero_weight_is_exact_main_loss():
    cfg = str(MODELS["c0"])
    auxiliary, plain = build_o2mc(cfg), build_model(cfg)
    plain.nc = 10
    auxiliary.aux_weight = 0
    auxiliary.train()
    plain.train()
    data = batch()
    torch.manual_seed(123)
    expected, expected_items = plain(data)
    torch.manual_seed(123)
    actual, items = auxiliary(data)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(items[:3], expected_items, rtol=0, atol=0)
    assert items[3:].sum() == 0


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(name + ": passed", flush=True)
