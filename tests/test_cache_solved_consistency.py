"""Regression: self.cache and self.solved must be cleared together.

If cache is retained across molecules while solved is cleared, a duplicate (or a
shared resolved intermediate) cache-hits req_search_tree and returns the cached
verdict without repopulating solved, so best_route truncates the node as a
non-stock leaf and reports it unresolved.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest
from unittest.mock import MagicMock

from shallowtree.chem.molecules.tree_molecule import TreeMolecule
from tests.search_helpers import _make_search


class TestCacheSolvedConsistency(unittest.TestCase):
    def _scenario(self):
        M = TreeMolecule(parent=None, smiles="c1ccccc1CO")
        A = TreeMolecule(parent=M, smiles="c1ccccc1")
        B = TreeMolecule(parent=M, smiles="OCCO")

        action_M = MagicMock()
        action_M.reactants = ((A, B),)
        action_M.metadata = {"classification": "cut1", "policy_name": "rules", "feasibility": 1.0}

        stock_inchis = {A.inchi_key, B.inchi_key}
        stock = MagicMock()
        stock.__contains__ = MagicMock(side_effect=lambda m: m.inchi_key in stock_inchis)

        exp = _make_search(stock=stock)

        def rules_side_effect(mols):
            return [action_M] if mols[0].inchi_key == M.inchi_key else []

        exp.rules_expansion.get_actions = MagicMock(side_effect=rules_side_effect)
        return exp, M

    def test_duplicate_molecule_both_resolve_with_same_route(self):
        exp, M = self._scenario()
        df = exp.search([M.smiles, M.smiles])
        self.assertEqual(df["resolved"].tolist(), [True, True])
        self.assertEqual(df["BBs"].tolist()[0], df["BBs"].tolist()[1])
        self.assertGreaterEqual(len(df["BBs"].tolist()[0]), 2)


if __name__ == "__main__":
    unittest.main()
