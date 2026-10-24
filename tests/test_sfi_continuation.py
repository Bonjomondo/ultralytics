"""Check convergence criteria and checkpoint preservation without a dataset."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.r18_p2_sfi.continue_train import ContinuationTrainer, plateau_status
from experiments.r18_p2_sfi.train import SFITrainer


class ContinuationTests(unittest.TestCase):
    def test_plateau_needs_budget_and_both_conditions(self):
        self.assertFalse(plateau_status([0.2] * 149)["converged"])
        self.assertTrue(plateau_status([0.2] * 150)["converged"])
        self.assertFalse(plateau_status([0.1 + i * 0.0001 for i in range(150)])["converged"])
        self.assertFalse(plateau_status([0.2] * 100 + [0.25] * 50)["converged"])
        self.assertFalse(plateau_status([0.2] * 130 + [0.19] * 20)["converged"])
        with self.assertRaises(FloatingPointError):
            plateau_status([float("nan")] * 150)

    def test_final_eval_keeps_raw_checkpoint_before_stripping(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            last = directory / "last.pt"
            last.write_bytes(b"raw checkpoint with optimizer")
            trainer = SimpleNamespace(last=last, wdir=directory)
            with patch.object(SFITrainer, "final_eval", lambda self: self.last.write_bytes(b"stripped")):
                # super() requires a real subclass instance; initialize no trainer machinery.
                actual = object.__new__(ContinuationTrainer)
                actual.last, actual.wdir = trainer.last, trainer.wdir
                actual.final_eval()
            self.assertEqual((directory / "resume.pt").read_bytes(), b"raw checkpoint with optimizer")
            self.assertEqual(last.read_bytes(), b"stripped")


if __name__ == "__main__":
    unittest.main()
