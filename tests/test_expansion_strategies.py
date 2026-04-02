"""Tests for expansion strategy cutoff logic."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from shallowtree.configs.expansion_configuration import ExpansionConfiguration


class TestCutoffPredictions(unittest.TestCase):
    """Test _cutoff_predictions logic."""

    def _make_strategy(self, cutoff_cumulative=0.995, cutoff_number=50, mask=None):
        """Create a TemplateBasedExpansionStrategy with mocked model and templates."""
        with patch("shallowtree.context.expansion_strategies.template_based_expansion_strategy.load_model") as mock_load, \
             patch("shallowtree.context.expansion_strategies.template_based_expansion_strategy.pd") as mock_pd:
            mock_model = MagicMock()
            mock_model.output_size = 100
            mock_load.return_value = mock_model

            mock_templates = MagicMock()
            mock_templates.__len__ = MagicMock(return_value=100)
            mock_pd.read_hdf.return_value = mock_templates

            config = ExpansionConfiguration(
                model="dummy_model",
                template="dummy.hdf5",
                cutoff_number=cutoff_number,
                cutoff_cumulative=cutoff_cumulative,
            )

            from shallowtree.context.expansion_strategies.template_based_expansion_strategy import (
                TemplateBasedExpansionStrategy,
            )
            strategy = TemplateBasedExpansionStrategy(key="test", config=config)
            strategy.mask = mask
            return strategy

    def test_cumulative_probability_cutoff(self):
        strategy = self._make_strategy(cutoff_cumulative=0.5, cutoff_number=100)
        predictions = np.ones(100) / 100
        result = strategy._cutoff_predictions(predictions)
        self.assertLessEqual(len(result), 51)

    def test_max_count_cutoff(self):
        strategy = self._make_strategy(cutoff_cumulative=0.999, cutoff_number=5)
        predictions = np.random.rand(100)
        predictions /= predictions.sum()
        result = strategy._cutoff_predictions(predictions)
        self.assertLessEqual(len(result), 5)

    def test_at_least_one_guard(self):
        strategy = self._make_strategy(cutoff_cumulative=0.0, cutoff_number=0)
        predictions = np.ones(100) / 100
        result = strategy._cutoff_predictions(predictions)
        self.assertGreaterEqual(len(result), 1)

    def test_mask_zeroes_excluded(self):
        mask = np.ones(100, dtype=bool)
        mask[0] = False
        strategy = self._make_strategy(cutoff_number=100, mask=mask)
        predictions = np.zeros(100)
        predictions[0] = 0.99
        predictions[1] = 0.01
        result = strategy._cutoff_predictions(predictions)
        self.assertNotIn(0, result[:1])


if __name__ == "__main__":
    unittest.main()
