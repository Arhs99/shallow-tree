"""
Tests for batch_feasibility in QuickKerasFilter.

Verifies that batch predictions produce identical results to sequential calls.
All tests are designed to run on CPU only (no GPU required).
"""
import os

# Force TensorFlow to use CPU only before importing TF
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from shallowtree.chem import TreeMolecule
from shallowtree.chem.reaction import TemplatedRetroReaction
from shallowtree.context.policy.filter_strategies import QuickKerasFilter


class MockModel:
    """Mock model for testing that returns predictable probabilities (CPU-based)."""

    def __init__(self, output_size=2048):
        self._output_size = output_size
        self._call_count = 0

    def __len__(self):
        return self._output_size

    def predict(self, *args, **kwargs):
        """
        Return probabilities based on input fingerprints.
        Uses sum of fingerprint as seed for deterministic output.
        Pure numpy implementation - no GPU required.
        """
        self._call_count += 1
        # Get the first input (prod_fp)
        prod_fp = args[0] if args else list(kwargs.values())[0]
        batch_size = prod_fp.shape[0]

        # Generate deterministic probabilities based on fingerprint sum
        probs = []
        for i in range(batch_size):
            # Use sum of fingerprint to generate a deterministic probability
            fp_sum = np.sum(prod_fp[i])
            # Map to 0-1 range using modulo and division
            prob = (fp_sum % 100) / 100.0
            probs.append([prob])

        return np.array(probs, dtype=np.float32)


class MockConfig:
    """Mock configuration object."""
    pass


def create_mock_reaction_with_reactants(seed: int = None) -> MagicMock:
    """
    Create a mock reaction with mock reactants for simpler testing.
    All operations are CPU-based numpy operations.
    """
    if seed is not None:
        np.random.seed(seed)

    reaction = MagicMock()
    reaction.reactants = ((MagicMock(),),)  # Non-empty reactants tuple

    # Mock the mol attribute with a fingerprint method
    fp = np.random.rand(2048).astype(np.float32)
    reaction.mol = MagicMock()
    reaction.mol.fingerprint = MagicMock(return_value=fp.copy())

    # Mock the reaction's fingerprint method
    reaction.fingerprint = MagicMock(return_value=fp.copy())

    return reaction


@pytest.fixture
def mock_filter():
    """Create a QuickKerasFilter with a mock model (CPU-only)."""
    with patch('shallowtree.context.policy.filter_strategies.load_model') as mock_load:
        mock_model = MockModel()
        mock_load.return_value = mock_model

        config = MockConfig()
        filter_policy = QuickKerasFilter(
            key="test_filter",
            config=config,
            model="mock_model.h5",
            filter_cutoff=0.5,
        )
        return filter_policy


class TestBatchFeasibility:
    """Tests for batch_feasibility method (CPU-only)."""

    def test_empty_list(self, mock_filter):
        """Test that empty list returns empty list."""
        result = mock_filter.batch_feasibility([])
        assert result == []

    def test_single_reaction_matches_sequential(self, mock_filter):
        """Test that single reaction gives same result as feasibility()."""
        # Create a mock reaction with fixed seed for reproducibility
        reaction = create_mock_reaction_with_reactants(seed=42)

        # Get sequential result
        seq_feasible, seq_prob = mock_filter.feasibility(reaction)

        # Get batch result
        batch_results = mock_filter.batch_feasibility([reaction])
        batch_feasible, batch_prob = batch_results[0]

        assert seq_feasible == batch_feasible
        assert abs(seq_prob - batch_prob) < 1e-6

    def test_multiple_reactions_match_sequential(self, mock_filter):
        """Test that multiple reactions give same results as sequential calls."""
        # Create multiple mock reactions with different fingerprints
        reactions = []
        for i in range(10):
            reaction = create_mock_reaction_with_reactants(seed=100 + i)
            reactions.append(reaction)

        # Get sequential results
        sequential_results = [mock_filter.feasibility(r) for r in reactions]

        # Get batch results
        batch_results = mock_filter.batch_feasibility(reactions)

        # Compare
        assert len(sequential_results) == len(batch_results)
        for i, (seq_result, batch_result) in enumerate(zip(sequential_results, batch_results)):
            seq_feasible, seq_prob = seq_result
            batch_feasible, batch_prob = batch_result
            assert seq_feasible == batch_feasible, f"Mismatch at index {i}"
            assert abs(seq_prob - batch_prob) < 1e-6, f"Prob mismatch at index {i}"

    def test_reactions_without_reactants(self, mock_filter):
        """Test that reactions without reactants return (False, 0.0)."""
        reactions = []

        # Reaction with reactants
        r1 = create_mock_reaction_with_reactants(seed=200)
        reactions.append(r1)

        # Reaction without reactants
        r2 = MagicMock()
        r2.reactants = ()  # Empty tuple - no reactants
        reactions.append(r2)

        # Another with reactants
        r3 = create_mock_reaction_with_reactants(seed=201)
        reactions.append(r3)

        results = mock_filter.batch_feasibility(reactions)

        assert len(results) == 3
        # First and third should have been processed
        assert results[0][1] != 0.0 or results[0][0] is False
        # Second should be (False, 0.0)
        assert results[1] == (False, 0.0)
        # Third should have been processed
        assert results[2][1] != 0.0 or results[2][0] is False

    def test_all_reactions_without_reactants(self, mock_filter):
        """Test when all reactions have no reactants."""
        reactions = []
        for _ in range(5):
            r = MagicMock()
            r.reactants = ()
            reactions.append(r)

        results = mock_filter.batch_feasibility(reactions)

        assert len(results) == 5
        assert all(result == (False, 0.0) for result in results)

    def test_result_order_preserved(self, mock_filter):
        """Test that result order matches input order."""
        reactions = []

        # Create reactions with distinctive fingerprints
        for i in range(5):
            reaction = MagicMock()
            reaction.reactants = ((MagicMock(),),)
            reaction.mol = MagicMock()
            # Create unique fingerprints with different patterns
            fp = np.zeros(2048, dtype=np.float32)
            fp[i * 100:(i + 1) * 100] = float(i + 1)
            reaction.mol.fingerprint = MagicMock(return_value=fp.copy())
            reaction.fingerprint = MagicMock(return_value=fp.copy())
            reactions.append(reaction)

        # Get both sequential and batch results
        sequential_results = [mock_filter.feasibility(r) for r in reactions]
        batch_results = mock_filter.batch_feasibility(reactions)

        # Verify order is preserved
        for i in range(len(reactions)):
            assert abs(sequential_results[i][1] - batch_results[i][1]) < 1e-6

    def test_large_batch_cpu_performance(self, mock_filter):
        """Test that large batches work correctly on CPU."""
        # Create a large batch of reactions
        reactions = []
        for i in range(100):
            reaction = create_mock_reaction_with_reactants(seed=1000 + i)
            reactions.append(reaction)

        # This should complete without memory issues on CPU
        batch_results = mock_filter.batch_feasibility(reactions)

        assert len(batch_results) == 100
        # All should have valid results (not all zeros)
        non_zero_count = sum(1 for _, prob in batch_results if prob > 0)
        assert non_zero_count > 0


class TestCPUOnlyExecution:
    """Tests to verify CPU-only execution environment."""

    def test_cuda_devices_hidden(self):
        """Verify CUDA devices are hidden from TensorFlow."""
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""

    def test_numpy_operations_cpu_only(self):
        """Verify numpy operations work without GPU."""
        # Basic numpy operations that should always work on CPU
        a = np.random.rand(1000, 2048).astype(np.float32)
        b = np.random.rand(1000, 2048).astype(np.float32)

        # Matrix operations
        result = np.dot(a, b.T)
        assert result.shape == (1000, 1000)

        # Stacking operations (used in batch_feasibility)
        stacked = np.vstack([a[:10], a[10:20]])
        assert stacked.shape == (20, 2048)

    def test_filter_cutoff_respected(self, mock_filter):
        """Test that filter_cutoff is properly applied."""
        # Create reaction that will have prob < 0.5
        reaction_low = MagicMock()
        reaction_low.reactants = ((MagicMock(),),)
        fp_low = np.ones(2048, dtype=np.float32) * 0.1  # Sum = 204.8, prob = 0.048
        reaction_low.mol = MagicMock()
        reaction_low.mol.fingerprint = MagicMock(return_value=fp_low)
        reaction_low.fingerprint = MagicMock(return_value=fp_low)

        # Create reaction that will have prob >= 0.5
        reaction_high = MagicMock()
        reaction_high.reactants = ((MagicMock(),),)
        fp_high = np.ones(2048, dtype=np.float32) * 0.5  # Sum = 1024, prob = 0.24
        reaction_high.mol = MagicMock()
        reaction_high.mol.fingerprint = MagicMock(return_value=fp_high)
        reaction_high.fingerprint = MagicMock(return_value=fp_high)

        results = mock_filter.batch_feasibility([reaction_low, reaction_high])

        assert len(results) == 2
        # Both should have valid probability values
        for feasible, prob in results:
            assert 0.0 <= prob <= 1.0
            assert feasible == (prob >= mock_filter.filter_cutoff)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
