"""Regression coverage for sorting and confidence filtering of RT-DETR queries."""
from types import SimpleNamespace

import unittest
import torch

from ultralytics.models.rtdetr.val import RTDETRValidator


def test_confidence_filter_preserves_sorted_query_identity(wrapped, threshold, expected):
    # Coordinates and class IDs identify queries independently of their scores.
    raw = torch.tensor([[[0.1, 0.5, 0.1, 0.2, 0.1, 0.0],
                         [0.3, 0.5, 0.1, 0.2, 0.0, 0.9],
                         [0.5, 0.5, 0.1, 0.2, 0.01, 0.0],
                         [0.7, 0.5, 0.1, 0.2, 0.8, 0.0]]])
    validator = object.__new__(RTDETRValidator)
    validator.args = SimpleNamespace(imgsz=100, conf=threshold)
    result = validator.postprocess((raw.clone(), None) if wrapped else raw.clone())[0]
    assert len(result["conf"]) == len(expected)
    torch.testing.assert_close(result["conf"], raw[0, expected, 4:].amax(-1))
    torch.testing.assert_close(result["cls"], raw[0, expected, 4:].argmax(-1).float())
    torch.testing.assert_close(result["bboxes"][:, 0], (raw[0, expected, 0] - raw[0, expected, 2] / 2) * 100)
    assert bool((result["conf"] > threshold).all())


def test_confidence_threshold_is_strict_and_batch_masks_are_independent():
    raw = torch.tensor([[[0.5, 0.5, 0.1, 0.1, 0.5], [0.5, 0.5, 0.1, 0.1, 0.9]],
                        [[0.5, 0.5, 0.1, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1, 0.2]]])
    validator = object.__new__(RTDETRValidator)
    validator.args = SimpleNamespace(imgsz=640, conf=0.5)
    result = validator.postprocess(raw)
    torch.testing.assert_close(result[0]["conf"], torch.tensor([0.9]))
    assert result[1]["bboxes"].shape == (0, 4)


class ConfidenceFilterTests(unittest.TestCase):
    def test_audit_measures_the_legacy_query_mismatch(self):
        from experiments.r18_p2_sfi.evaluate import ConfidenceAuditValidator

        validator = object.__new__(ConfidenceAuditValidator)
        validator.args = SimpleNamespace(imgsz=100, conf=0.5)
        raw = torch.tensor([[[0.1, 0.5, 0.1, 0.2, 0.1],
                             [0.3, 0.5, 0.1, 0.2, 0.9],
                             [0.5, 0.5, 0.1, 0.2, 0.01],
                             [0.7, 0.5, 0.1, 0.2, 0.8]]])
        result = validator.postprocess(raw)
        self.assertEqual(validator.pending_audit[0]["filtered_queries"], 2)
        self.assertEqual(validator.pending_audit[0]["wrongly_removed_above_threshold"], 1)
        self.assertEqual(validator.pending_audit[0]["wrongly_retained_below_threshold"], 1)
        torch.testing.assert_close(result[0]["conf"], torch.tensor([0.9, 0.8]))

    def test_query_identity(self):
        for wrapped in (False, True):
            for threshold, expected in [(0.5, [1, 3]), (0.0, [1, 3, 0, 2]), (1.0, [])]:
                with self.subTest(wrapped=wrapped, threshold=threshold):
                    test_confidence_filter_preserves_sorted_query_identity(wrapped, threshold, expected)

    def test_batch_and_threshold(self):
        test_confidence_threshold_is_strict_and_batch_masks_are_independent()


if __name__ == "__main__":
    unittest.main()
