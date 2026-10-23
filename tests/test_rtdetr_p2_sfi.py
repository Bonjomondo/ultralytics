"""P2 branch semantics and full RT-DETR integration checks (CPU, no data downloads)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.r18_p2_sfi.train import MODELS, RECIPE, build_model
from ultralytics import RTDETR
from ultralytics.nn.modules import SPDConv, SPDLite, SpaceToDepth2


class SFITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_space_to_depth_keeps_all_pixels_in_documented_order(self):
        x = torch.arange(16.0).reshape(1, 1, 4, 4).requires_grad_()
        actual = SpaceToDepth2()(x)
        expected = torch.tensor([[[[0, 2], [8, 10]], [[4, 6], [12, 14]],
                                  [[1, 3], [9, 11]], [[5, 7], [13, 15]]]], dtype=x.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        with self.assertRaises(ValueError):
            SpaceToDepth2()(torch.zeros(1, 1, 3, 4))

    def test_lite_convolution_budget(self):
        spd, lite = SPDConv(64, 128), SPDLite(64, 128, 64)
        count = lambda module: sum(m.weight.numel() for m in module.modules() if isinstance(m, torch.nn.Conv2d))
        self.assertEqual(count(spd), 294912)
        self.assertEqual(count(lite), 25152)
        self.assertLess(count(lite) / count(spd), 0.086)

    def test_full_models_shapes_shared_init_backward_and_checkpoint(self):
        baseline = build_model(str(MODELS["c0"]))
        reference = baseline.state_dict()
        shared_hash = baseline.ablation_initialization["shared_state_sha256"]
        for variant in ("c1", "c2", "c3"):
            with self.subTest(variant=variant):
                model = build_model(str(MODELS[variant])).eval()
                self.assertEqual(model.ablation_initialization["shared_state_sha256"], shared_hash)
                for name, value in reference.items():
                    if not name.startswith("model.14."):
                        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
                head = model.model[-1]
                self.assertEqual(head.f, [16, 23, 22])
                self.assertEqual(head.num_decoder_layers, 3)
                received = []
                handle = head.register_forward_pre_hook(lambda module, inputs: received.extend(t.shape for t in inputs[0]))
                with torch.no_grad(), patch("torch.fft.fft2", side_effect=AssertionError("FFT forbidden")), \
                        patch("torch.fft.ifft2", side_effect=AssertionError("IFFT forbidden")):
                    self.assertEqual(model(torch.rand(1, 3, 128, 128))[0].shape, (1, 300, 14))
                handle.remove()
                self.assertEqual(received, [(1, 256, 16, 16), (1, 256, 8, 8), (1, 256, 4, 4)])
                model.train()
                model.nc = 10
                loss, parts = model({"img": torch.rand(2, 3, 128, 128),
                    "batch_idx": torch.tensor([0, 1]), "cls": torch.tensor([[0.0], [3.0]]),
                    "bboxes": torch.tensor([[0.3, 0.4, 0.1, 0.2], [0.6, 0.5, 0.2, 0.1]])})
                self.assertTrue(torch.isfinite(loss).all() and torch.isfinite(parts).all())
                loss.sum().backward()
                grad = next(model.model[14].parameters()).grad
                self.assertIsNotNone(grad)
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(), 0)
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "checkpoint.pt"
                    torch.save({"model": model.eval()}, path)
                    restored = RTDETR(str(path)).model
                    self.assertEqual(restored.yaml["head"], model.yaml["head"])

    def test_recipe_matches_existing_r18(self):
        self.assertEqual((RECIPE["batch"], RECIPE["nbs"], RECIPE["warmup_epochs"]), (8, 64, 3.0))
        self.assertFalse(RECIPE["amp"])
        self.assertEqual(RECIPE["warmup_bias_lr"], 0.0)


if __name__ == "__main__":
    unittest.main()
