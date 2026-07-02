"""Tests for parallel iterative deepening (execution_modes).

Two layers:
- Fast unit tests for the orchestration (longest-first dispatch + input-order
  restore), the per-target try/except sentinel, and the heavy-atom cost proxy,
  with the pool and the worker search mocked out — no models/stock/Redis needed.
- An integration parity test (real models + stock + Redis, like
  test_execution_modes) asserting parallel output equals the sequential
  iterative-deepening output per target.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd

import shallowtree.interfaces.execution_modes as em
from shallowtree.configs.input_configuration import InputConfiguration

_EM = "shallowtree.interfaces.execution_modes"
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "application_config/config.json"


class TestHeavyAtomCount(unittest.TestCase):
    def test_counts_heavy_atoms_only(self):
        self.assertEqual(em._heavy_atom_count("C"), 1)        # methane, H's excluded
        self.assertEqual(em._heavy_atom_count("CCO"), 3)
        self.assertEqual(em._heavy_atom_count("c1ccccc1"), 6)

    def test_unparseable_sorts_last(self):
        self.assertEqual(em._heavy_atom_count("not-a-smiles"), 0)


class TestRunOneTargetSentinel(unittest.TestCase):
    """_run_one_target isolates a failing target and reuses the warm instance."""

    def setUp(self):
        em._WORKER.clear()
        self.addCleanup(em._WORKER.clear)

    def _seed_worker(self, search):
        em._WORKER["search"] = search

    def test_success_row_carries_error_none(self):
        search = MagicMock()
        search.search_iterative.return_value = pd.DataFrame([{
            "SMILES": "CCO", "score": 1.0, "resolved": True, "route": {},
            "BBs": ["CCO"], "resolved_depth": 2,
        }])
        self._seed_worker(search)
        cfg = InputConfiguration(app_configuration_path="x.json", output_path="",
                                 smiles=["CCO"])
        row = em._run_one_target(cfg)
        self.assertTrue(row["resolved"])
        self.assertEqual(row["resolved_depth"], 2)
        self.assertIsNone(row["error"])
        search.search_iterative.assert_called_once_with(["CCO"])

    def test_throwing_target_returns_uniform_sentinel(self):
        search = MagicMock()
        search.search_iterative.side_effect = RuntimeError("boom")
        self._seed_worker(search)
        cfg = InputConfiguration(app_configuration_path="x.json", output_path="",
                                 smiles=["BAD"])
        row = em._run_one_target(cfg)
        # Same columns a normal row carries, so pd.concat never NaN-pads.
        self.assertEqual(set(row), {"SMILES", "score", "resolved", "route",
                                    "BBs", "resolved_depth", "error"})
        self.assertFalse(row["resolved"])
        self.assertIsNone(row["resolved_depth"])
        self.assertEqual(row["error"], "boom")
        self.assertEqual(row["SMILES"], "BAD")


class TestParallelOrchestration(unittest.TestCase):
    """Longest-first dispatch + input-order restoration, pool/worker mocked."""

    def setUp(self):
        em._WORKER.clear()
        self.addCleanup(em._WORKER.clear)

    def _run(self, smiles):
        dispatched = []

        def fake_target(task):
            smi = task.smiles[0]
            dispatched.append(smi)
            return {"SMILES": smi, "score": 1.0, "resolved": True, "route": {},
                    "BBs": [], "resolved_depth": 2, "error": None}

        shared = MagicMock()
        shared.shm_name = "shm"
        shared.__len__.return_value = 5

        pool = MagicMock()
        pool.map.side_effect = lambda fn, tasks: [fn(t) for t in tasks]

        cfg = InputConfiguration(app_configuration_path="x.json", output_path="",
                                 smiles=smiles, iterative_deepening=True,
                                 parallel_processes=2, d_start=2, depth=3)

        with patch(f"{_EM}.Configuration.from_json", return_value={}), \
             patch(f"{_EM}.ApplicationConfiguration") as app_cls, \
             patch(f"{_EM}._build_shared_stock", return_value=shared), \
             patch(f"{_EM}._build_worker_stock", return_value=None), \
             patch(f"{_EM}.ProcessPool", return_value=pool), \
             patch(f"{_EM}._run_one_target", side_effect=fake_target):
            app_cls.return_value = MagicMock()
            df = em.parallel_iterative_deepening_search(cfg)
        return df, dispatched, shared

    def test_dispatch_is_longest_first_and_output_restores_input_order(self):
        # HAC: C=1, c1ccccc1=6, CCO=3, CCCCCCCC=8
        smiles = ["C", "c1ccccc1", "CCO", "CCCCCCCC"]
        df, dispatched, shared = self._run(smiles)
        # Costliest target dispatched first, cheapest last.
        self.assertEqual(dispatched, ["CCCCCCCC", "c1ccccc1", "CCO", "C"])
        # Result frame restored to the ORIGINAL input order.
        self.assertEqual(df["SMILES"].tolist(), smiles)
        # Shared stock unlinked exactly once (the finally).
        shared.unlink.assert_called_once()

    def test_route_dropped_when_not_routes(self):
        smiles = ["CCO", "C"]
        dispatched = []

        def fake_target(task):
            dispatched.append(task.smiles[0])
            return {"SMILES": task.smiles[0], "score": 1.0, "resolved": True, "route": {},
                    "BBs": [], "resolved_depth": 2, "error": None}

        shared = MagicMock(); shared.shm_name = "shm"; shared.__len__.return_value = 5
        pool = MagicMock(); pool.map.side_effect = lambda fn, tasks: [fn(t) for t in tasks]
        cfg = InputConfiguration(app_configuration_path="x.json", output_path="",
                                 smiles=smiles, iterative_deepening=True,
                                 parallel_processes=2, d_start=2, depth=3, routes=False)
        with patch(f"{_EM}.Configuration.from_json", return_value={}), \
             patch(f"{_EM}.ApplicationConfiguration", return_value=MagicMock()), \
             patch(f"{_EM}._build_shared_stock", return_value=shared), \
             patch(f"{_EM}._build_worker_stock", return_value=None), \
             patch(f"{_EM}.ProcessPool", return_value=pool), \
             patch(f"{_EM}._run_one_target", side_effect=fake_target):
            df = em.parallel_iterative_deepening_search(cfg)
        self.assertNotIn("route", df.columns)


class TestParallelIddfsParity(unittest.TestCase):
    """Integration: parallel IDDFS == sequential IDDFS per target.

    Requires the real ONNX models, the stock, and a running Redis (mirrors the
    existing test_execution_modes integration tests)."""

    def setUp(self):
        config = json.loads(CONFIG_PATH.read_text())
        config.setdefault("cache", {})["enabled"] = True
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(config, tmp)
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))

        self.smiles = [
            "Clc1ccccc1COC5CC(Nc3n[nH]c4cc(c2ccccc2)ccc34)C5",
            "CC(C)(C)c1cc2c(N/N=C\\c3cccc(CN)n3)ncnc2s1",
            "COc1cccc2c1c(Cl)c1c3c(cc(O)c(O)c32)C(=O)N1",
        ]
        self.config = InputConfiguration(
            app_configuration_path=tmp.name, scaffold=None, routes=True,
            depth=3, smiles=self.smiles, output_path="",
            iterative_deepening=True, d_start=2, d_max=3, parallel_processes=3)

    def test_parallel_matches_sequential(self):
        seq = em.iterative_deepening_search(self.config).reset_index(drop=True)
        par = em.parallel_iterative_deepening_search(self.config).reset_index(drop=True)

        # Same targets, same order.
        self.assertEqual(par["SMILES"].tolist(), seq["SMILES"].tolist())
        self.assertEqual(par["resolved"].tolist(), seq["resolved"].tolist())
        self.assertEqual(par["BBs"].tolist(), seq["BBs"].tolist())
        # resolved_depth is NaN for unresolved targets; fill so NaN==NaN compares.
        self.assertEqual(par["resolved_depth"].fillna(-1).tolist(),
                         seq["resolved_depth"].fillna(-1).tolist())


if __name__ == "__main__":
    unittest.main()
