"""Tests for the iterative-deepening driver (BaseTreeSearch.search_iterative)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest
from unittest.mock import MagicMock

import pandas as pd

from shallowtree.chem.molecules.tree_molecule import TreeMolecule
from tests.search_helpers import _make_search


def _row_df(resolved, score):
    """One-row DataFrame shaped like StandardSearch.search output."""
    return pd.DataFrame([{
        "SMILES": "X", "score": score, "resolved": resolved, "route": {}, "BBs": [],
    }])


class TestSearchIterativeDriver(unittest.TestCase):
    """Driver control flow, with the per-depth search mocked out."""

    def test_stops_at_first_resolved_depth(self):
        exp = _make_search()
        # Unresolved at d=2,3; resolved at d=4. Extra entries should never be used.
        exp.search = MagicMock(side_effect=[
            _row_df(False, 0.3), _row_df(False, 0.6), _row_df(True, 1.0),
            _row_df(True, 1.0),
        ])
        exp._input_config.d_start = 2
        exp._input_config.d_max = 10
        df = exp.search_iterative(["X"])
        self.assertEqual(exp.search.call_count, 3)  # 2, 3, 4 then stop
        self.assertEqual(df.iloc[0]["resolved_depth"], 4)
        self.assertTrue(bool(df.iloc[0]["resolved"]))
        # The instance depth was left at the resolving depth.
        self.assertEqual(exp._input_config.depth, 4)

    def test_returns_none_when_unresolved_within_d_max(self):
        exp = _make_search()
        exp.search = MagicMock(side_effect=[_row_df(False, 0.1) for _ in range(4)])
        exp._input_config.d_start = 2
        exp._input_config.d_max = 5
        df = exp.search_iterative(["X"])
        self.assertEqual(exp.search.call_count, 4)  # 2, 3, 4, 5
        self.assertIsNone(df.iloc[0]["resolved_depth"])
        self.assertFalse(bool(df.iloc[0]["resolved"]))
        self.assertEqual(exp._input_config.depth, 5)  # swept to d_max

    def test_resolves_on_first_depth(self):
        exp = _make_search()
        exp.search = MagicMock(side_effect=[_row_df(True, 1.0)])
        exp._input_config.d_start = 2
        exp._input_config.d_max = 10
        df = exp.search_iterative(["X"])
        self.assertEqual(exp.search.call_count, 1)
        self.assertEqual(df.iloc[0]["resolved_depth"], 2)

    def test_multiple_targets_each_get_own_resolved_depth(self):
        exp = _make_search()
        # target 1 resolves at d=2 (1 call); target 2 resolves at d=3 (2 calls).
        exp.search = MagicMock(side_effect=[
            _row_df(True, 1.0),                       # t1 @ d=2
            _row_df(False, 0.4), _row_df(True, 1.0),  # t2 @ d=2, d=3
        ])
        exp._input_config.d_start = 2
        exp._input_config.d_max = 5
        df = exp.search_iterative(["A", "B"])
        self.assertEqual(list(df["resolved_depth"]), [2, 3])


class TestSearchIterativeEndToEnd(unittest.TestCase):
    """Real req_search_tree + budget cache across a warm deepening sweep."""

    def _scenario(self):
        # M => A + B ; B => C + D. A, C, D in stock; B is not.
        # At max_depth 1, B's children fall past the depth boundary -> B unresolved
        # -> M unresolved. At max_depth 2, C/D are reached in-budget and in stock
        # -> B resolved -> M resolved. So the minimal resolving depth is 2.
        M = TreeMolecule(parent=None, smiles="c1ccccc1CO")
        A = TreeMolecule(parent=M, smiles="c1ccccc1")
        B = TreeMolecule(parent=M, smiles="OCCO")
        C = TreeMolecule(parent=B, smiles="CO")
        D = TreeMolecule(parent=B, smiles="OCO")

        action_M = MagicMock()
        action_M.reactants = ((A, B),)
        action_M.metadata = {"classification": "cut1", "policy_name": "rules", "feasibility": 1.0}
        action_B = MagicMock()
        action_B.reactants = ((C, D),)
        action_B.metadata = {"classification": "cut2", "policy_name": "rules", "feasibility": 1.0}

        stock_inchis = {A.inchi_key, C.inchi_key, D.inchi_key}
        stock = MagicMock()
        stock.__contains__ = MagicMock(side_effect=lambda m: m.inchi_key in stock_inchis)

        exp = _make_search(stock=stock)

        def rules_side_effect(mols):
            ik = mols[0].inchi_key
            if ik == M.inchi_key:
                return [action_M]
            if ik == B.inchi_key:
                return [action_B]
            return []

        exp.rules_expansion.get_actions = MagicMock(side_effect=rules_side_effect)
        return exp, M, B

    def test_finds_minimal_resolving_depth(self):
        exp, M, _ = self._scenario()
        exp._input_config.d_start = 1
        exp._input_config.d_max = 4
        df = exp.search_iterative([M.smiles])
        self.assertEqual(df.iloc[0]["resolved_depth"], 2)
        self.assertTrue(bool(df.iloc[0]["resolved"]))

    def test_stock_leaves_never_expanded_during_sweep(self):
        # Stock molecules short-circuit before expansion; only M and B are ever
        # expanded across the whole sweep — no redundant work on settled leaves.
        exp, M, B = self._scenario()
        exp._input_config.d_start = 1
        exp._input_config.d_max = 4
        exp.search_iterative([M.smiles])
        expanded = {call.args[0][0].inchi_key for call in exp.rules_expansion.get_actions.call_args_list}
        self.assertEqual(expanded, {M.inchi_key, B.inchi_key})


if __name__ == "__main__":
    unittest.main()
