"""Tests for shallowtree.interfaces.full_tree_search — Expander tree search."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch

from shallowtree.chem.mol import TreeMolecule


def _make_expander(**overrides):
    """Create an Expander with mocked dependencies."""
    with patch("shallowtree.interfaces.full_tree_search.Expander._setup_filter_policy") as mock_fp, \
         patch("shallowtree.interfaces.full_tree_search.Expander._setup_expansion_policy") as mock_ep, \
         patch("shallowtree.interfaces.full_tree_search.Expander._setup_stock") as mock_st, \
         patch("shallowtree.interfaces.full_tree_search.Expander._setup_redis_cache") as mock_rc, \
         patch("shallowtree.interfaces.full_tree_search.Expander._setup_rules_expansion") as mock_re:

        mock_stock = MagicMock()
        mock_stock.__contains__ = MagicMock(return_value=False)

        mock_expansion = MagicMock()
        mock_expansion.get_actions = MagicMock(return_value=([], []))

        mock_rules = MagicMock()
        mock_rules.get_actions = MagicMock(return_value=[])

        mock_filter = MagicMock()

        mock_fp.return_value = mock_filter
        mock_ep.return_value = mock_expansion
        mock_st.return_value = mock_stock
        mock_rc.return_value = None
        mock_re.return_value = mock_rules

        mock_config = MagicMock()
        from shallowtree.interfaces.full_tree_search import Expander
        exp = Expander(mock_config)

        # Apply overrides
        for k, v in overrides.items():
            setattr(exp, k, v)

        return exp


def _make_action(reactant_smiles_list, classification="test", policy_name="ml",
                 feasibility=1.0):
    """Create a mock action with real TreeMolecule reactants."""
    parent = TreeMolecule(parent=None, smiles="CCO")
    reactants = tuple(
        TreeMolecule(parent=parent, smiles=smi) for smi in reactant_smiles_list
    )
    action = MagicMock()
    action.reactants = (reactants,)
    action.metadata = {
        "classification": classification,
        "policy_name": policy_name,
        "feasibility": feasibility,
    }
    return action


class TestReqSearchTree(unittest.TestCase):
    """Test req_search_tree logic."""

    def test_stock_mol_returns_1(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)
        mol = TreeMolecule(parent=None, smiles="CCO")
        score = exp.req_search_tree(mol, depth=0)
        self.assertEqual(score, 1.0)

    def test_stock_mol_cached_at_depth_0(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)
        mol = TreeMolecule(parent=None, smiles="CCO")
        exp.req_search_tree(mol, depth=0)
        self.assertIn(mol.inchi_key, exp.cache)
        self.assertEqual(exp.cache[mol.inchi_key], (0, 1.0))

    def test_depth_exceeds_max_returns_0(self):
        exp = _make_expander()
        exp.max_depth = 2
        mol = TreeMolecule(parent=None, smiles="CCO")
        score = exp.req_search_tree(mol, depth=3)
        self.assertEqual(score, 0.0)

    def test_cache_hit_reuse_when_cdepth_le_depth(self):
        exp = _make_expander()
        mol = TreeMolecule(parent=None, smiles="CCO")
        exp.cache[mol.inchi_key] = (1, 0.8)
        score = exp.req_search_tree(mol, depth=2)
        self.assertEqual(score, 0.8)

    def test_cache_skip_when_cdepth_gt_depth(self):
        """When cached depth > current depth, cache should be skipped."""
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)
        mol = TreeMolecule(parent=None, smiles="CCO")
        exp.cache[mol.inchi_key] = (2, 0.5)
        score = exp.req_search_tree(mol, depth=0)
        self.assertEqual(score, 1.0)

    def test_mean_of_reactants_scoring(self):
        """Score = mean of reactant scores."""
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)
        exp.max_depth = 2

        mol = TreeMolecule(parent=None, smiles="c1ccccc1CO")
        action = _make_action(["c1ccccc1", "CO"], policy_name="rules")

        exp.expansion_policy.get_actions = MagicMock(return_value=([], []))
        exp.rules_expansion.get_actions = MagicMock(return_value=[action])

        score = exp.req_search_tree(mol, depth=0)
        self.assertEqual(score, 1.0)

    def test_solved_threshold(self):
        """Only scores > 0.9 are added to self.solved."""
        mol = TreeMolecule(parent=None, smiles="c1ccccc1CO")
        reactant1 = TreeMolecule(parent=mol, smiles="c1ccccc1")
        reactant2 = TreeMolecule(parent=mol, smiles="CO")

        stock_inchis = {reactant1.inchi_key, reactant2.inchi_key}
        stock = MagicMock()
        stock.__contains__ = MagicMock(side_effect=lambda m: m.inchi_key in stock_inchis)
        exp = _make_expander(stock=stock)
        exp.max_depth = 2

        action = MagicMock()
        action.reactants = ((reactant1, reactant2),)
        action.metadata = {"classification": "test", "policy_name": "rules", "feasibility": 1.0}

        exp.expansion_policy.get_actions = MagicMock(return_value=([], []))
        exp.rules_expansion.get_actions = MagicMock(return_value=[action])

        exp.req_search_tree(mol, depth=0)
        self.assertIn(mol.inchi_key, exp.solved)

    def test_filter_threshold_gates_actions(self):
        """ML actions with feasibility < 0.5 should be excluded."""
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=False)
        exp = _make_expander(stock=stock)
        exp.max_depth = 2

        mol = TreeMolecule(parent=None, smiles="c1ccccc1CO")
        action = _make_action(["c1ccccc1", "CO"], policy_name="ml_policy")

        exp.expansion_policy.get_actions = MagicMock(return_value=([action], [0.5]))
        exp.rules_expansion.get_actions = MagicMock(return_value=[])

        mock_filter = MagicMock()
        mock_filter.batch_feasibility = MagicMock(return_value=[("", 0.3)])
        exp.filter_policy.__getitem__ = MagicMock(return_value=mock_filter)
        exp.filter_policy.selection = {"test_filter": True}

        score = exp.req_search_tree(mol, depth=0)
        self.assertEqual(score, 0.0)

    def test_rules_actions_bypass_filter(self):
        """Actions with policy_name='rules' bypass filter check."""
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)
        exp.max_depth = 2

        mol = TreeMolecule(parent=None, smiles="c1ccccc1CO")
        action = _make_action(["c1ccccc1", "CO"], policy_name="rules")

        exp.expansion_policy.get_actions = MagicMock(return_value=([], []))
        exp.rules_expansion.get_actions = MagicMock(return_value=[action])

        score = exp.req_search_tree(mol, depth=0)
        self.assertGreater(score, 0.9)


class TestBestRoute(unittest.TestCase):
    """Test best_route reconstruction."""

    def test_single_step_reconstruction(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)
        exp.max_depth = 2

        mol = TreeMolecule(parent=None, smiles="c1ccccc1CO")
        reactant1 = TreeMolecule(parent=mol, smiles="c1ccccc1")
        reactant2 = TreeMolecule(parent=mol, smiles="CO")
        exp.solved[mol.inchi_key] = ((reactant1, reactant2), 1.0, "test")

        tree = defaultdict(list)
        exp.BBs = []
        exp.best_route(mol, 0, tree)

        self.assertIn(1, tree)
        self.assertEqual(len(exp.BBs), 2)

    def test_unsolved_mol_becomes_bb(self):
        exp = _make_expander()
        exp.max_depth = 2

        mol = TreeMolecule(parent=None, smiles="CCO")
        tree = defaultdict(list)
        exp.BBs = []
        exp.best_route(mol, 0, tree)

        self.assertIn("CCO", exp.BBs)
        self.assertEqual(len(tree), 0)


class TestSearchTree(unittest.TestCase):
    """Test search_tree returns correct DataFrame."""

    def test_returns_dataframe_with_correct_columns(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)

        df = exp.search_tree(["CCO"], max_depth=2)
        self.assertIn("SMILES", df.columns)
        self.assertIn("score", df.columns)
        self.assertIn("route", df.columns)
        self.assertIn("BBs", df.columns)

    def test_handles_multiple_smiles(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)

        df = exp.search_tree(["CCO", "CCCO"], max_depth=2)
        self.assertEqual(len(df), 2)

    def test_works_with_no_redis_cache(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)
        exp.redis_cache = None

        df = exp.search_tree(["CCO"], max_depth=2)
        self.assertEqual(len(df), 1)


class TestCacheCorrectness(unittest.TestCase):
    """Test cache behavior."""

    def test_cache_populated_after_search(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)

        mol = TreeMolecule(parent=None, smiles="CCO")
        exp.req_search_tree(mol, depth=0)
        self.assertIn(mol.inchi_key, exp.cache)

    def test_depth_stored_correctly(self):
        mol = TreeMolecule(parent=None, smiles="c1ccccc1CO")
        reactant1 = TreeMolecule(parent=mol, smiles="c1ccccc1")
        reactant2 = TreeMolecule(parent=mol, smiles="CO")

        stock_inchis = {reactant1.inchi_key, reactant2.inchi_key}
        stock = MagicMock()
        stock.__contains__ = MagicMock(side_effect=lambda m: m.inchi_key in stock_inchis)
        exp = _make_expander(stock=stock)
        exp.max_depth = 2

        action = MagicMock()
        action.reactants = ((reactant1, reactant2),)
        action.metadata = {"classification": "test", "policy_name": "rules", "feasibility": 1.0}
        exp.expansion_policy.get_actions = MagicMock(return_value=([], []))
        exp.rules_expansion.get_actions = MagicMock(return_value=[action])

        exp.req_search_tree(mol, depth=1)
        depth, _ = exp.cache[mol.inchi_key]
        self.assertEqual(depth, 1)


if __name__ == "__main__":
    unittest.main()
