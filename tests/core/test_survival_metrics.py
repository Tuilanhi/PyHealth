"""Tests for survival discrimination metrics."""

import importlib.util
import math
import unittest

import numpy as np
import torch

from pyhealth.metrics import concordance_index

HAS_LIFELINES = importlib.util.find_spec("lifelines") is not None


class TestConcordanceIndex(unittest.TestCase):
    def test_perfect_and_reversed_ranking(self):
        duration, event = [1.0, 2.0, 3.0, 4.0], [1, 1, 1, 0]
        self.assertEqual(concordance_index([4, 3, 2, 1], duration, event), 1.0)
        self.assertEqual(concordance_index([1, 2, 3, 4], duration, event), 0.0)

    def test_hand_computed_with_censoring(self):
        # Comparable pairs (i had the event, j outlived it):
        # (0,1) (0,2) (0,3) (2,3). Risks 0.5, 0.1, 0.9, 0.2, so (0,1), (0,3),
        # and (2,3) are concordant and (0,2) is not.
        risk = [0.5, 0.1, 0.9, 0.2]
        duration = [1.0, 2.0, 3.0, 4.0]
        event = [1, 0, 1, 0]
        self.assertAlmostEqual(concordance_index(risk, duration, event), 3 / 4)

    def test_tied_risks(self):
        risk, duration, event = [1.0, 1.0], [1.0, 2.0], [1, 0]
        self.assertEqual(concordance_index(risk, duration, event), 0.5)
        self.assertEqual(
            concordance_index(risk, duration, event, tied_risk="zero"), 0.0
        )

    def test_tied_times(self):
        # Two deaths at the same time are not comparable; a death and a
        # censoring at the same time are (the censored one outlived the death).
        self.assertTrue(math.isnan(concordance_index([2, 1], [1.0, 1.0], [1, 1])))
        self.assertEqual(concordance_index([2, 1], [1.0, 1.0], [1, 0]), 1.0)

    def test_no_comparable_pairs_returns_nan(self):
        self.assertTrue(math.isnan(concordance_index([1, 2], [1.0, 2.0], [0, 0])))

    def test_accepts_tensors_and_column_vectors(self):
        risk = torch.tensor([[3.0], [2.0], [1.0]])
        self.assertEqual(
            concordance_index(risk, torch.tensor([1.0, 2.0, 3.0]), np.array([1, 1, 0])),
            1.0,
        )

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            concordance_index([1, 2], [1.0, 2.0, 3.0], [1, 0, 1])
        with self.assertRaises(ValueError):
            concordance_index([1, 2], [1.0, 2.0], [1, 2])
        with self.assertRaises(ValueError):
            concordance_index([1, float("nan")], [1.0, 2.0], [1, 0])
        with self.assertRaises(ValueError):
            concordance_index([1, 2], [1.0, 2.0], [1, 0], tied_risk="third")

    @unittest.skipUnless(HAS_LIFELINES, "lifelines is not installed")
    def test_matches_lifelines_with_ties(self):
        from lifelines.utils import concordance_index as lifelines_cindex

        rng = np.random.default_rng(0)
        for _ in range(20):
            n = int(rng.integers(5, 40))
            duration = rng.integers(1, 8, n).astype(float)
            event = rng.integers(0, 2, n)
            event[0] = 1
            risk = rng.integers(0, 5, n).astype(float)
            try:
                expected = lifelines_cindex(duration, -risk, event)
            except ZeroDivisionError:
                continue
            self.assertAlmostEqual(concordance_index(risk, duration, event), expected)


if __name__ == "__main__":
    unittest.main()
