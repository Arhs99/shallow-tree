"""Tests for StandardSearch.search — DataFrame output shape."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest
from unittest.mock import MagicMock

from tests.search_helpers import _make_search


class TestSearch(unittest.TestCase):
    """Test StandardSearch.search returns correct DataFrame."""

    def test_returns_dataframe_with_correct_columns(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(stock=stock)

        df = exp.search(["CCO"], max_depth=2)
        self.assertIn("SMILES", df.columns)
        self.assertIn("score", df.columns)
        self.assertIn("route", df.columns)
        self.assertIn("BBs", df.columns)

    def test_handles_multiple_smiles(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(stock=stock)

        df = exp.search(["CCO", "CCCO"], max_depth=2)
        self.assertEqual(len(df), 2)

    def test_works_with_no_redis_cache(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(stock=stock)
        exp.redis_cache = None

        df = exp.search(["CCO"], max_depth=2)
        self.assertEqual(len(df), 1)


if __name__ == "__main__":
    unittest.main()
