"""Tests for BaseTreeSearch.req_search_tree."""
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest
from unittest.mock import MagicMock

from shallowtree.chem.molecules.tree_molecule import TreeMolecule
from tests.search_helpers import _make_search, _make_action


class TestReqSearchTree(unittest.TestCase):
    """Test req_search_tree logic. Returns (score, resolved)."""

    def test_stock_mol_returns_1(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(stock=stock)
        mol = TreeMolecule(parent=None, smiles="CCO")
        score, resolved = exp.req_search_tree(mol, depth=0, start_time=time.time())
        self.assertEqual(score, 1.0)
        self.assertTrue(resolved)

    def test_stock_mol_cached_as_budget_0_always_reusable(self):
        # A stock leaf is cached as (budget=0, 1.0, True). The True reuse rule
        # (budget_now >= 0, always true) makes it reusable at any budget.
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(stock=stock)
        exp._input_config.depth = 5
        mol = TreeMolecule(parent=None, smiles="CCO")
        exp.req_search_tree(mol, depth=0, start_time=time.time())
        self.assertIn(mol.inchi_key, exp.cache)
        self.assertEqual(exp.cache[mol.inchi_key], (0, 1.0, True))
        # Reused even at the deepest query (smallest budget); stock check would
        # also re-confirm, so flip stock off to prove the cache entry is used.
        stock.__contains__ = MagicMock(return_value=False)
        score, resolved = exp.req_search_tree(mol, depth=5, start_time=time.time())
        self.assertEqual((score, resolved), (1.0, True))

    def test_depth_exceeds_max_returns_0(self):
        exp = _make_search()
        exp._input_config.depth = 2
        mol = TreeMolecule(parent=None, smiles="CCO")
        score, resolved = exp.req_search_tree(mol, depth=3, start_time=time.time())
        self.assertEqual(score, 0.0)
        self.assertFalse(resolved)

    def test_resolved_reused_when_budget_now_ge_cached(self):
        # resolved=True at budget K_c=2 is valid at any larger budget. Query at
        # depth 0 with max_depth 4 -> budget_now=4 >= 2 -> reuse.
        exp = _make_search()
        exp._input_config.depth = 4
        mol = TreeMolecule(parent=None, smiles="CCO")
        exp.cache[mol.inchi_key] = (2, 1.0, True)
        score, resolved = exp.req_search_tree(mol, depth=0, start_time=time.time())
        self.assertEqual((score, resolved), (1.0, True))

    def test_resolved_recomputed_when_budget_now_lt_cached(self):
        # resolved=True at budget K_c=4 must NOT be reused at a smaller budget
        # (the route may not fit). depth 3, max_depth 4 -> budget_now=1 < 4 ->
        # recompute; with the mol now in stock the fresh verdict is (1.0, True).
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(stock=stock)
        exp._input_config.depth = 4
        mol = TreeMolecule(parent=None, smiles="CCO")
        exp.cache[mol.inchi_key] = (4, 0.7, True)
        score, resolved = exp.req_search_tree(mol, depth=3, start_time=time.time())
        self.assertEqual((score, resolved), (1.0, True))

    def test_unresolved_reused_when_budget_now_le_cached(self):
        # resolved=False at budget K_c=3 is valid at any smaller-or-equal budget.
        # depth 2, max_depth 4 -> budget_now=2 <= 3 -> reuse the False verdict.
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(stock=stock)
        exp._input_config.depth = 4
        mol = TreeMolecule(parent=None, smiles="CCO")
        exp.cache[mol.inchi_key] = (3, 0.5, False)
        score, resolved = exp.req_search_tree(mol, depth=2, start_time=time.time())
        self.assertEqual((score, resolved), (0.5, False))

    def test_unresolved_recomputed_when_budget_now_gt_cached(self):
        # resolved=False at budget K_c=1 must NOT be reused at a larger budget
        # (the extra room may resolve it). depth 0, max_depth 4 -> budget_now=4 > 1
        # -> recompute; with the mol now in stock the fresh verdict is (1.0, True).
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(stock=stock)
        exp._input_config.depth = 4
        mol = TreeMolecule(parent=None, smiles="CCO")
        exp.cache[mol.inchi_key] = (1, 0.5, False)
        score, resolved = exp.req_search_tree(mol, depth=0, start_time=time.time())
        self.assertEqual((score, resolved), (1.0, True))

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

        score, resolved = exp.req_search_tree(mol, depth=0, start_time=time.time())
        self.assertEqual(score, 1.0)
        self.assertTrue(resolved)

    def test_resolved_route_is_solved(self):
        """A route whose reactants are all in stock is resolved and solved."""
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

        score, resolved = exp.req_search_tree(mol, depth=0, start_time=time.time())
        self.assertTrue(resolved)
        self.assertIn(mol.inchi_key, exp.solved)

    def test_unresolved_route_not_solved(self):
        """A route with a non-stock leaf is NOT resolved and NOT committed to solved,
        even when the soft score crosses the acceptance threshold."""
        mol = TreeMolecule(parent=None, smiles="c1ccccc1CO")
        reactant1 = TreeMolecule(parent=mol, smiles="c1ccccc1")
        reactant2 = TreeMolecule(parent=mol, smiles="CO")

        # Only one of the two leaves is in stock -> route is not fully resolved.
        stock_inchis = {reactant1.inchi_key}
        stock = MagicMock()
        stock.__contains__ = MagicMock(side_effect=lambda m: m.inchi_key in stock_inchis)
        exp = _make_search(stock=stock)
        exp._input_config.depth = 1  # reactant2 cannot be expanded further -> dead leaf

        action = MagicMock()
        action.reactants = ((reactant1, reactant2),)
        action.metadata = {"classification": "test", "policy_name": "rules", "feasibility": 1.0}

        exp.expansion_policy.get_actions = MagicMock(return_value=([], []))
        exp.rules_expansion.get_actions = MagicMock(return_value=[action])

        score, resolved = exp.req_search_tree(mol, depth=0, start_time=time.time())
        self.assertFalse(resolved)
        self.assertNotIn(mol.inchi_key, exp.solved)

    def test_cycle_guard_breaks_self_referential_path(self):
        """A reactant that reappears on its own path scores 0.0/unresolved and is
        not cached (the verdict is path-dependent)."""
        mol = TreeMolecule(parent=None, smiles="c1ccccc1CO")
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=False)
        exp = _make_search(stock=stock)
        exp._input_config.depth = 5

        score, resolved = exp.req_search_tree(mol, depth=0, start_time=time.time(), ancestors=frozenset({mol.inchi_key}))
        self.assertEqual(score, 0.0)
        self.assertFalse(resolved)
        self.assertNotIn(mol.inchi_key, exp.cache)

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

        score, resolved = exp.req_search_tree(mol, depth=0, start_time=time.time())
        self.assertEqual(score, 0.0)
        self.assertFalse(resolved)

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

        score, resolved = exp.req_search_tree(mol, depth=0, start_time=time.time())
        self.assertGreater(score, 0.9)
        self.assertTrue(resolved)


if __name__ == "__main__":
    unittest.main()
