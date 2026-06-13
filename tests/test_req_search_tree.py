"""Tests for BaseTreeSearch.req_search_tree."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest
from unittest.mock import MagicMock

from shallowtree.chem.molecules.tree_molecule import TreeMolecule
from tests.search_helpers import _make_search, _make_action


class TestReqSearchTree(unittest.TestCase):
    """Test req_search_tree logic."""

    def test_stock_mol_returns_1(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(stock=stock)
        mol = TreeMolecule(parent=None, smiles="CCO")
        score = exp.req_search_tree(mol, depth=0)
        self.assertEqual(score, 1.0)

    def test_stock_mol_cached_at_depth_0(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(stock=stock)
        mol = TreeMolecule(parent=None, smiles="CCO")
        exp.req_search_tree(mol, depth=0)
        self.assertIn(mol.inchi_key, exp.cache)
        self.assertEqual(exp.cache[mol.inchi_key], (0, 1.0))

    def test_depth_exceeds_max_returns_0(self):
        exp = _make_search()
        exp._input_config.depth = 2
        mol = TreeMolecule(parent=None, smiles="CCO")
        score = exp.req_search_tree(mol, depth=3)
        self.assertEqual(score, 0.0)

    def test_cache_hit_reuse_when_cdepth_le_depth(self):
        exp = _make_search()
        mol = TreeMolecule(parent=None, smiles="CCO")
        exp.cache[mol.inchi_key] = (1, 0.8)
        score = exp.req_search_tree(mol, depth=2)
        self.assertEqual(score, 0.8)

    def test_cache_skip_when_cdepth_gt_depth(self):
        """When cached depth > current depth, cache should be skipped."""
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(stock=stock)
        mol = TreeMolecule(parent=None, smiles="CCO")
        exp.cache[mol.inchi_key] = (2, 0.5)
        score = exp.req_search_tree(mol, depth=0)
        self.assertEqual(score, 1.0)

    def test_mean_of_reactants_scoring(self):
        """Score = mean of reactant scores."""
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(stock=stock)
        exp._input_config.depth = 2

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
        exp = _make_search(stock=stock)
        exp._input_config.depth = 2

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
        exp = _make_search(stock=stock)
        exp._input_config.depth = 2

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
        exp = _make_search(stock=stock)
        exp._input_config.depth = 2

        mol = TreeMolecule(parent=None, smiles="c1ccccc1CO")
        action = _make_action(["c1ccccc1", "CO"], policy_name="rules")

        exp.expansion_policy.get_actions = MagicMock(return_value=([], []))
        exp.rules_expansion.get_actions = MagicMock(return_value=[action])

        score = exp.req_search_tree(mol, depth=0)
        self.assertGreater(score, 0.9)


if __name__ == "__main__":
    unittest.main()
