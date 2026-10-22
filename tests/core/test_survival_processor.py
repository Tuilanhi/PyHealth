"""Tests for survival label and standardized feature processors."""

import unittest

import torch

from pyhealth.processors import (
    PROCESSOR_REGISTRY,
    StandardizedTensorProcessor,
    SurvivalIntervalProcessor,
    SurvivalLabelProcessor,
)

NAN = float("nan")


class TestSurvivalIntervalProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = SurvivalIntervalProcessor()

    def test_registered(self):
        self.assertIs(
            PROCESSOR_REGISTRY["survival_interval"], SurvivalIntervalProcessor
        )

    def test_process_returns_float_tensor(self):
        out = self.processor.process([0.5, 1.25, 1])
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.shape, (3,))
        torch.testing.assert_close(out, torch.tensor([0.5, 1.25, 1.0]))

    def test_accepts_tensor_input(self):
        out = self.processor.process(torch.tensor([0.0, 2.0, 0.0]))
        torch.testing.assert_close(out, torch.tensor([0.0, 2.0, 0.0]))

    def test_rejects_malformed_intervals(self):
        for bad in (
            [1.0, 1.0, 0],
            [2.0, 1.0, 0],
            [-0.1, 1.0, 0],
            [0.0, 1.0, 2],
            [0.0, 1.0],
            [0.0, NAN, 1],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.processor.process(bad)

    def test_metadata(self):
        self.assertEqual(self.processor.size(), 3)
        self.assertFalse(self.processor.is_token())
        self.assertEqual(self.processor.dim(), (1,))
        self.assertTrue(self.processor.stores_tensor)


class TestSurvivalLabelProcessor(unittest.TestCase):
    def test_process_and_validation(self):
        processor = SurvivalLabelProcessor()
        self.assertIs(PROCESSOR_REGISTRY["survival"], SurvivalLabelProcessor)
        torch.testing.assert_close(
            processor.process([1.5, 1]), torch.tensor([1.5, 1.0])
        )
        for bad in ([0.0, 1], [-1.0, 0], [1.0, 0.5], [1.0, 0, 0]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                processor.process(bad)
        self.assertEqual(processor.size(), 2)


class TestStandardizedTensorProcessor(unittest.TestCase):
    def test_registered_and_learns_statistics(self):
        self.assertIs(
            PROCESSOR_REGISTRY["standardized_tensor"], StandardizedTensorProcessor
        )
        self.assertTrue(StandardizedTensorProcessor.learns_statistics)
        self.assertTrue(StandardizedTensorProcessor.stores_tensor)

    def test_zscores_with_fitted_statistics(self):
        processor = StandardizedTensorProcessor()
        processor.fit([{"x": [1.0, 10.0]}, {"x": [3.0, 30.0]}], "x")
        torch.testing.assert_close(
            processor.process([3.0, 10.0]), torch.tensor([1.0, -1.0])
        )
        self.assertEqual(processor.size(), 2)
        self.assertEqual(processor.dim(), (1,))

    def test_statistics_come_only_from_fit_samples(self):
        processor = StandardizedTensorProcessor()
        processor.fit([{"x": [0.0]}, {"x": [2.0]}], "x")
        # A test-time outlier is transformed, not used to refit.
        torch.testing.assert_close(processor.process([101.0]), torch.tensor([100.0]))

    def test_missing_values_ignored_in_fit_and_filled_with_mean(self):
        processor = StandardizedTensorProcessor()
        processor.fit([{"x": [1.0, NAN]}, {"x": [3.0, 4.0]}, {"x": [NAN, 6.0]}], "x")
        torch.testing.assert_close(
            processor.process([NAN, NAN]), torch.tensor([0.0, 0.0])
        )
        torch.testing.assert_close(
            processor.process([1.0, 6.0]), torch.tensor([-1.0, 1.0])
        )

    def test_constant_feature_is_centered_only(self):
        processor = StandardizedTensorProcessor()
        processor.fit([{"x": [5.0]}, {"x": [5.0]}], "x")
        torch.testing.assert_close(processor.process([7.0]), torch.tensor([2.0]))

    def test_sequence_input(self):
        processor = StandardizedTensorProcessor()
        processor.fit([{"x": [[0.0], [2.0]]}, {"x": [[4.0]]}], "x")
        out = processor.process([[2.0], [NAN]])
        self.assertEqual(out.shape, (2, 1))
        torch.testing.assert_close(out, torch.tensor([[0.0], [0.0]]))
        self.assertEqual(processor.spatial(), (True, False))

    def test_errors(self):
        processor = StandardizedTensorProcessor()
        with self.assertRaises(RuntimeError):
            processor.process([1.0])
        with self.assertRaises(ValueError):
            processor.fit([{"y": [1.0]}], "x")
        with self.assertRaises(ValueError):
            processor.fit([{"x": [1.0]}, {"x": [1.0, 2.0]}], "x")
        processor.fit([{"x": [1.0, 2.0]}], "x")
        with self.assertRaises(ValueError):
            processor.process([1.0])


if __name__ == "__main__":
    unittest.main()
