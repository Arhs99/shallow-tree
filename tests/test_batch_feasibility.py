"""
Tests for batch_feasibility in QuickKerasFilter.

Verifies that batch predictions produce identical results to sequential calls.
All tests are designed to run on CPU only (no GPU required).
"""
import os

# Force TensorFlow to use CPU only before importing TF
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from shallowtree.configs.filter_configuration import FilterConfiguration
from shallowtree.context.filters.quick_keras_filter import QuickKerasFilter


class MockModel:
    """Mock model for testing that returns predictable probabilities (CPU-based)."""

    def __init__(self, output_size=2048):
        self._output_size = output_size
        self._call_count = 0

    def __len__(self):
        return self._output_size

    def predict(self, *args, **kwargs):
        self._call_count += 1
        prod_fp = args[0] if args else list(kwargs.values())[0]
        batch_size = prod_fp.shape[0]

        probs = []
        for i in range(batch_size):
            fp_sum = np.sum(prod_fp[i])
            prob = (fp_sum % 100) / 100.0
            probs.append([prob])

        return np.array(probs, dtype=np.float32)


def create_mock_reaction(seed=None):
    """Create a mock reaction with mock reactants."""
    if seed is not None:
        np.random.seed(seed)

    reaction = MagicMock()
    reaction.reactants = ((MagicMock(),),)

    fp = np.random.rand(2048).astype(np.float32)
    reaction.mol = MagicMock()
    reaction.mol.fingerprint = MagicMock(return_value=fp.copy())
    reaction.fingerprint = MagicMock(return_value=fp.copy())

    return reaction


def _make_filter():
    """Create a QuickKerasFilter with a mock model (CPU-only)."""
    with patch('shallowtree.context.filters.quick_keras_filter.load_model') as mock_load:
        mock_load.return_value = MockModel()

        config = FilterConfiguration(
            model="mock_model.h5",
            filter_cutoff=0.5,
        )
        return QuickKerasFilter(key="test_filter", config=config)


class TestBatchFeasibility(unittest.TestCase):
    """Tests for batch_feasibility method (CPU-only)."""

    def setUp(self):
        self.filter = _make_filter()

    def test_empty_list(self):
        result = self.filter.batch_feasibility([])
        self.assertEqual(result, [])

    def test_single_reaction_matches_sequential(self):
        reaction = create_mock_reaction(seed=42)

        seq_feasible, seq_prob = self.filter.feasibility(reaction)
        batch_results = self.filter.batch_feasibility([reaction])
        batch_feasible, batch_prob = batch_results[0]

        self.assertEqual(seq_feasible, batch_feasible)
        self.assertAlmostEqual(seq_prob, batch_prob, places=5)

    def test_multiple_reactions_match_sequential(self):
        reactions = [create_mock_reaction(seed=100 + i) for i in range(10)]

        sequential_results = [self.filter.feasibility(r) for r in reactions]
        batch_results = self.filter.batch_feasibility(reactions)

        self.assertEqual(len(sequential_results), len(batch_results))
        for i, (seq, batch) in enumerate(zip(sequential_results, batch_results)):
            self.assertEqual(seq[0], batch[0], f"Mismatch at index {i}")
            self.assertAlmostEqual(seq[1], batch[1], places=5, msg=f"Prob mismatch at index {i}")

    def test_reactions_without_reactants(self):
        r1 = create_mock_reaction(seed=200)
        r2 = MagicMock()
        r2.reactants = ()
        r3 = create_mock_reaction(seed=201)

        results = self.filter.batch_feasibility([r1, r2, r3])

        self.assertEqual(len(results), 3)
        self.assertEqual(results[1], (False, 0.0))

    def test_all_reactions_without_reactants(self):
        reactions = []
        for _ in range(5):
            r = MagicMock()
            r.reactants = ()
            reactions.append(r)

        results = self.filter.batch_feasibility(reactions)

        self.assertEqual(len(results), 5)
        self.assertTrue(all(r == (False, 0.0) for r in results))

    def test_result_order_preserved(self):
        reactions = []
        for i in range(5):
            reaction = MagicMock()
            reaction.reactants = ((MagicMock(),),)
            reaction.mol = MagicMock()
            fp = np.zeros(2048, dtype=np.float32)
            fp[i * 100:(i + 1) * 100] = float(i + 1)
            reaction.mol.fingerprint = MagicMock(return_value=fp.copy())
            reaction.fingerprint = MagicMock(return_value=fp.copy())
            reactions.append(reaction)

        sequential_results = [self.filter.feasibility(r) for r in reactions]
        batch_results = self.filter.batch_feasibility(reactions)

        for i in range(len(reactions)):
            self.assertAlmostEqual(sequential_results[i][1], batch_results[i][1], places=5)

    def test_large_batch(self):
        reactions = [create_mock_reaction(seed=1000 + i) for i in range(100)]

        batch_results = self.filter.batch_feasibility(reactions)

        self.assertEqual(len(batch_results), 100)
        non_zero = sum(1 for _, prob in batch_results if prob > 0)
        self.assertGreater(non_zero, 0)

    def test_filter_cutoff_respected(self):
        reaction_low = MagicMock()
        reaction_low.reactants = ((MagicMock(),),)
        fp_low = np.ones(2048, dtype=np.float32) * 0.1
        reaction_low.mol = MagicMock()
        reaction_low.mol.fingerprint = MagicMock(return_value=fp_low)
        reaction_low.fingerprint = MagicMock(return_value=fp_low)

        reaction_high = MagicMock()
        reaction_high.reactants = ((MagicMock(),),)
        fp_high = np.ones(2048, dtype=np.float32) * 0.5
        reaction_high.mol = MagicMock()
        reaction_high.mol.fingerprint = MagicMock(return_value=fp_high)
        reaction_high.fingerprint = MagicMock(return_value=fp_high)

        results = self.filter.batch_feasibility([reaction_low, reaction_high])

        self.assertEqual(len(results), 2)
        for feasible, prob in results:
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
            self.assertEqual(feasible, prob >= self.filter.filter_cutoff)


if __name__ == "__main__":
    unittest.main()
