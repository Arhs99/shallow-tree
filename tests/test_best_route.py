"""Tests for StandardSearch.best_route reconstruction."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest
from collections import defaultdict
from unittest.mock import MagicMock

from shallowtree.chem.molecules.tree_molecule import TreeMolecule
from tests.search_helpers import _make_search


class TestBestRoute(unittest.TestCase):
    """Test best_route reconstruction (StandardSearch)."""

    def test_single_step_reconstruction(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(stock=stock)
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
        exp = _make_search()
        exp.max_depth = 2

        mol = TreeMolecule(parent=None, smiles="CCO")
        tree = defaultdict(list)
        exp.BBs = []
        exp.best_route(mol, 0, tree)

        self.assertIn("CCO", exp.BBs)
        self.assertEqual(len(tree), 0)

    def test_best_route_warns_when_solved_missing_for_non_stock_mol(self):
        """Guard against silent route truncation from LRU eviction or redis
        partial writes: a non-stock mol missing from self.solved should warn."""
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=False)
        exp = _make_search(stock=stock)
        exp.max_depth = 2
        exp.solved = {}

        mol = TreeMolecule(parent=None, smiles="CCO")
        tree = defaultdict(list)
        exp.BBs = []
        with self.assertLogs(exp._logger, level="WARNING") as cm:
            exp.best_route(mol, 0, tree)
        self.assertTrue(any("route truncated" in m for m in cm.output))
        self.assertIn("CCO", exp.BBs)

    def test_best_route_silent_for_stock_mol(self):
        """The normal stock-termination path stays silent (no warning)."""
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(stock=stock)
        exp.max_depth = 2
        exp.solved = {}

        mol = TreeMolecule(parent=None, smiles="CCO")
        tree = defaultdict(list)
        exp.BBs = []
        with self.assertNoLogs(exp._logger, level="WARNING"):
            exp.best_route(mol, 0, tree)
        self.assertIn("CCO", exp.BBs)

    def _make_boundary_solved(self, exp):
        """Build a solved chain root->mid->leaf->deep with the leaf reached at
        depth max_depth+1. Returns (root, leaf_smiles, deep_smiles)."""
        root = TreeMolecule(parent=None, smiles="c1ccccc1CO")
        mid = TreeMolecule(parent=root, smiles="c1ccccc1C(=O)O")
        leaf = TreeMolecule(parent=mid, smiles="CCO")
        deep = TreeMolecule(parent=leaf, smiles="CCN")
        exp.solved = {
            root.inchi_key: ((mid,), 1.0, "r1"),
            mid.inchi_key: ((leaf,), 1.0, "r2"),
            leaf.inchi_key: ((deep,), 1.0, "r3"),  # would expand if not depth-capped
        }
        return root, leaf.smiles, deep.smiles

    def test_best_route_records_depth_boundary_leaf(self):
        """A solved node reached past max_depth must be treated as a leaf:
        recorded in BBs, warned (non-stock), and never expanded further."""
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=False)
        exp = _make_search(stock=stock)
        exp.max_depth = 1
        root, leaf_smiles, deep_smiles = self._make_boundary_solved(exp)

        tree = defaultdict(list)
        exp.BBs = []
        with self.assertLogs(exp._logger, level="WARNING") as cm:
            exp.best_route(root, 0, tree)

        self.assertTrue(any("route truncated" in m for m in cm.output))
        self.assertIn(leaf_smiles, exp.BBs)       # boundary leaf recorded
        self.assertNotIn(deep_smiles, exp.BBs)    # not expanded past the limit
        self.assertNotIn(3, tree)                 # no depth-3 reaction emitted
        self.assertIn(2, tree)                    # mid => leaf still emitted

    def test_best_route_depth_boundary_leaf_silent_when_in_stock(self):
        """An in-stock boundary leaf is recorded but does not warn."""
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(stock=stock)
        exp.max_depth = 1
        root, leaf_smiles, deep_smiles = self._make_boundary_solved(exp)

        tree = defaultdict(list)
        exp.BBs = []
        with self.assertNoLogs(exp._logger, level="WARNING"):
            exp.best_route(root, 0, tree)

        self.assertIn(leaf_smiles, exp.BBs)
        self.assertNotIn(deep_smiles, exp.BBs)
        self.assertNotIn(3, tree)


if __name__ == "__main__":
    unittest.main()
