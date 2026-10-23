# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run with python -m unittest discover -s tests -p test_rtdetr_r18_na.py -v."""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from train_paper_repro_visdrone import parse_bool
from train_rtdetr_r18_na import MODEL_CONFIGS, R18AblationTrainer, build_ablation_model, parse_args
from ultralytics import RTDETR
from ultralytics.nn.modules import NeighborhoodAttention2d, ResNet18Stage, ResNet18Stem, ResidualLocalRefine
from ultralytics.nn.modules.rtdetr_na import ResNet18BasicBlock


class NeighborhoodAttentionTests(unittest.TestCase):
    """Check boundary semantics and the zero-initialized gate's learning behavior."""

    @classmethod
    def setUpClass(cls):
        """Avoid CPU oversubscription on shared hosts."""
        torch.set_num_threads(2)

    def test_attention_matches_valid_neighbor_reference(self):
        """Compare against explicitly enumerated valid neighbors, including image corners."""
        torch.manual_seed(5)
        module = NeighborhoodAttention2d(8, 2, 5, use_rpb=True).double()
        x = torch.randn(1, 8, 3, 4, dtype=torch.float64)
        q, k, v = (t.reshape(1, 2, 4, 3, 4) for t in module.qkv(x).chunk(3, dim=1))
        expected = torch.empty_like(q)
        for y in range(3):
            for col in range(4):
                keys, values, biases = [], [], []
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if 0 <= y + dy < 3 and 0 <= col + dx < 4:
                            keys.append(k[:, :, :, y + dy, col + dx])
                            values.append(v[:, :, :, y + dy, col + dx])
                            biases.append(module.relative_position_bias[:, (dy + 2) * 5 + dx + 2])
                scores = (q[:, :, :, y, col, None] * torch.stack(keys, -1)).sum(2) * module.scale
                scores = scores + torch.stack(biases, -1)[None]
                expected[:, :, :, y, col] = (scores.softmax(-1).unsqueeze(2) * torch.stack(values, -1)).sum(-1)
        torch.testing.assert_close(module(x), module.proj(expected.reshape_as(x)), rtol=1e-10, atol=1e-10)

    def test_gate_identity_then_attention_gradients(self):
        """Gamma learns first; attention receives gradients after the gate opens."""
        torch.manual_seed(3)
        module = ResidualLocalRefine(16, 8, 2, 5)
        self.assertIsNone(module.attn.relative_position_bias)
        x = torch.randn(2, 16, 5, 7, requires_grad=True)
        torch.testing.assert_close(module(x), x, rtol=0, atol=0)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        module(x).square().mean().backward()
        self.assertGreater(module.gamma.grad.abs().item(), 0)
        self.assertEqual(module.attn.qkv.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        module(x).square().mean().backward()
        self.assertTrue(torch.isfinite(module.attn.qkv.weight.grad).all())
        self.assertGreater(module.attn.qkv.weight.grad.abs().sum().item(), 0)

    def test_invalid_dimensions_and_windows(self):
        """Reject invalid channel/head counts and nonpositive or even windows."""
        for kwargs in (
            {"dim": 7}, {"dim": 8, "num_heads": 0}, {"dim": 0},
            {"dim": 8, "kernel_size": 0}, {"dim": 8, "kernel_size": 4}, {"dim": 8, "kernel_size": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NeighborhoodAttention2d(**kwargs)

    def test_backbone_blocks_and_odd_downsampling(self):
        """ResNet18 uses eight two-convolution blocks and matching stride-two shortcuts."""
        stem = ResNet18Stem(3).eval()
        x = stem(torch.randn(1, 3, 132, 132))
        self.assertEqual(x.shape, (1, 64, 33, 33))
        stages = nn.Sequential(
            *(ResNet18Stage(a, b, s) for a, b, s in (
                (64, 64, 1), (64, 128, 2), (128, 256, 2), (256, 512, 2)
            ))
        ).eval()
        self.assertEqual(sum(isinstance(m, ResNet18BasicBlock) for m in stages.modules()), 8)
        self.assertEqual(stages(x).shape, (1, 512, 5, 5))


class ModelIntegrationTests(unittest.TestCase):
    """Exercise YAML parsing, shared initialization, topology, loss, and serialization."""

    @classmethod
    def setUpClass(cls):
        """Build paired models with exactly matching common parameters."""
        torch.set_num_threads(2)
        cls.r0 = build_ablation_model(str(MODEL_CONFIGS["r0"])).eval()
        cls.r1 = build_ablation_model(str(MODEL_CONFIGS["r1"])).eval()

    def test_common_state_is_identical(self):
        """A same-seed pair must share decoder weights as well as backbone/neck weights."""
        other = self.r1.state_dict()
        for name, value in self.r0.state_dict().items():
            torch.testing.assert_close(value, other[name], rtol=0, atol=0)
        self.assertEqual(
            {k: v for k, v in self.r0.ablation_initialization.items() if k != "new_variant_state"},
            {k: v for k, v in self.r1.ablation_initialization.items() if k != "new_variant_state"},
        )
        self.assertEqual(self.r0.ablation_initialization["new_variant_state"], [])
        self.assertTrue(self.r1.ablation_initialization["new_variant_state"])
        self.assertEqual(self.r1.model[-1].num_decoder_layers, 3)
        self.assertEqual(self.r1.model[-1].f, [16, 23, 22])

    def test_only_decoder_p4_branch_changes(self):
        """Opening the gate changes the decoder's P4 input while preserving P3 and P5."""
        image = torch.randn(1, 3, 128, 128)
        received = {}

        def save_input(module, inputs):
            received["features"] = [t.clone() for t in inputs[0]]

        h0 = self.r0.model[-1].register_forward_pre_hook(save_input)
        with torch.no_grad():
            out0 = self.r0(image)[0]
            base_features = received["features"]
        h0.remove()
        h1 = self.r1.model[-1].register_forward_pre_hook(save_input)
        with torch.no_grad():
            out1 = self.r1(image)[0]
            torch.testing.assert_close(out0, out1, rtol=0, atol=0)
            self.r1.model[23].gamma.fill_(0.5)
            self.r1(image)
            changed_features = received["features"]
            self.r1.model[23].gamma.zero_()
        h1.remove()
        for index in (0, 2):
            torch.testing.assert_close(base_features[index], changed_features[index], rtol=0, atol=0)
        self.assertFalse(torch.equal(base_features[1], changed_features[1]))

    def test_detection_loss_and_backward(self):
        """Use the real Hungarian-matched detection loss, including denoising queries."""
        model = build_ablation_model(str(MODEL_CONFIGS["r1"])).train()
        model.nc = 10  # normally assigned by DetectionTrainer.set_model_attributes()
        batch = {
            "img": torch.rand(2, 3, 128, 128),
            "batch_idx": torch.tensor([0, 1]),
            "cls": torch.tensor([[0.0], [3.0]]),
            "bboxes": torch.tensor([[0.3, 0.4, 0.1, 0.2], [0.6, 0.5, 0.2, 0.1]]),
        }
        loss, components = model(batch)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(components).all())
        loss.sum().backward()
        self.assertIsNotNone(model.model[23].gamma.grad)
        self.assertTrue(torch.isfinite(model.model[23].gamma.grad).all())
        self.assertIsNotNone(model.model[-1].dec_bbox_head[-1].layers[-1].weight.grad)
        model.zero_grad(set_to_none=True)
        model.eval()

    def test_checkpoint_roundtrip(self):
        """Custom classes must remain loadable by the ordinary RTDETR inference API."""
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "r1.pt"
            torch.save({"model": self.r1}, checkpoint)
            restored = RTDETR(str(checkpoint)).model
            self.assertIsInstance(restored.model[23], ResidualLocalRefine)
            self.assertEqual(restored.model[-1].num_decoder_layers, 3)

    def test_trainer_uses_paired_initialization(self):
        """Check the trainer rebuild path that ordinary model.train() actually calls."""
        from types import SimpleNamespace

        trainer = object.__new__(R18AblationTrainer)
        trainer.data = {"nc": 10, "channels": 3}
        trainer.args = SimpleNamespace(seed=0)
        model = trainer.get_model(str(MODEL_CONFIGS["r1"]), verbose=False)
        self.assertEqual(model.ablation_initialization, self.r1.ablation_initialization)


class ArgumentTests(unittest.TestCase):
    """Verify experiment defaults and unambiguous command-line booleans."""

    def test_defaults_and_amp_false(self):
        """R1's explicit --amp False must stay false."""
        args = parse_args(["--amp", "False"])
        self.assertFalse(args.amp)
        self.assertEqual((args.variant, args.epochs, args.batch, args.patience), ("r1", 150, 8, 0))
        self.assertEqual(args.warmup_bias_lr, 0.0)
        self.assertTrue(parse_bool("true"))
        self.assertFalse(parse_bool("0"))


if __name__ == "__main__":
    unittest.main()
