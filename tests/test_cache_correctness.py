"""Tests for the in-memory branch cache populated during search."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time

import unittest
from unittest.mock import MagicMock

from shallowtree.chem.molecules.tree_molecule import TreeMolecule
from tests.search_helpers import _make_search


class TestCacheCorrectness(unittest.TestCase):
    """Test cache behavior."""

    def test_cache_populated_after_search(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(stock=stock)

        mol = TreeMolecule(parent=None, smiles="CCO")
        exp.req_search_tree(mol, depth=0, start_time=time.time())
        self.assertIn(mol.inchi_key, exp.cache)

    def test_budget_stored_correctly(self):
        # The cache stores the remaining BUDGET (max_depth - depth) at which the
        # verdict held, not the absolute tree-depth. With max_depth=3 and a query
        # at depth=1 the stored budget must be 2 (distinct from the depth, 1).
        mol = TreeMolecule(parent=None, smiles="c1ccccc1CO")
        reactant1 = TreeMolecule(parent=mol, smiles="c1ccccc1")
        reactant2 = TreeMolecule(parent=mol, smiles="CO")

        stock_inchis = {reactant1.inchi_key, reactant2.inchi_key}
        stock = MagicMock()
        stock.__contains__ = MagicMock(side_effect=lambda m: m.inchi_key in stock_inchis)
        exp = _make_search(stock=stock)
        exp._input_config.depth = 3

        action = MagicMock()
        action.reactants = ((reactant1, reactant2),)
        action.metadata = {"classification": "test", "policy_name": "rules", "feasibility": 1.0}
        exp.expansion_policy.get_actions = MagicMock(return_value=([], []))
        exp.rules_expansion.get_actions = MagicMock(return_value=[action])

        exp.req_search_tree(mol, depth=1, start_time=time.time())
        budget, _, _ = exp.cache[mol.inchi_key]
        self.assertEqual(budget, 2)


if __name__ == "__main__":
    unittest.main()
