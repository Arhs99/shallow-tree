"""Tests for InMemoryInchiKeyQuery and PackedInchiKeySet."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock

import pandas as pd

from shallowtree.context.stock.packed_inchi_key_set import PackedInchiKeySet
from shallowtree.context.stock.queries import InMemoryInchiKeyQuery
from shallowtree.utils.exceptions import StockException


# 27-char ASCII InChI-key-shaped strings used across tests.
KEY_A = "AAAAAAAAAAAAAA-BBBBBBBBBB-C"
KEY_B = "DDDDDDDDDDDDDD-EEEEEEEEEE-F"
KEY_C = "GGGGGGGGGGGGGG-HHHHHHHHHH-I"


class TestPackedInchiKeySet(unittest.TestCase):
    def test_membership_hits_and_misses(self):
        s = PackedInchiKeySet.from_iterable([KEY_B, KEY_A, KEY_C])
        self.assertIn(KEY_A, s)
        self.assertIn(KEY_B, s)
        self.assertIn(KEY_C, s)
        self.assertNotIn("ZZZZZZZZZZZZZZ-ZZZZZZZZZZ-Z", s)

    def test_dedup_and_sort(self):
        s = PackedInchiKeySet.from_iterable([KEY_C, KEY_A, KEY_B, KEY_A])
        self.assertEqual(len(s), 3)
        self.assertEqual(list(s), sorted([KEY_A, KEY_B, KEY_C]))

    def test_accepts_bytes_input(self):
        s = PackedInchiKeySet.from_iterable([KEY_A.encode("ascii")])
        self.assertIn(KEY_A, s)
        self.assertIn(KEY_A.encode("ascii"), s)

    def test_rejects_wrong_length_and_non_ascii(self):
        s = PackedInchiKeySet.from_iterable([KEY_A])
        self.assertNotIn("short", s)
        self.assertNotIn("X" * 28, s)
        self.assertNotIn("café_" + "X" * 22, s)
        self.assertNotIn(12345, s)

    def test_raw_buffer_layout(self):
        s = PackedInchiKeySet.from_iterable([KEY_B, KEY_A])
        buf = bytes(s.raw_buffer())
        self.assertEqual(len(buf), 2 * 27)
        self.assertEqual(buf[:27].decode("ascii"), KEY_A)
        self.assertEqual(buf[27:].decode("ascii"), KEY_B)


def _make_mol_with_inchi(key):
    mol = MagicMock()
    mol.inchi_key = key
    return mol


class TestInMemoryInchiKeyQueryHDF5(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.h5_path = os.path.join(self.tmpdir, "stock.h5")
        df = pd.DataFrame({"inchi_key": [KEY_A, KEY_B, KEY_C]})
        df.to_hdf(self.h5_path, key="table")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_loads_from_hdf5(self):
        q = InMemoryInchiKeyQuery(self.h5_path)
        self.assertEqual(len(q), 3)
        self.assertIn(_make_mol_with_inchi(KEY_A), q)
        self.assertIn(_make_mol_with_inchi(KEY_B), q)
        self.assertIn(_make_mol_with_inchi(KEY_C), q)
        self.assertNotIn(_make_mol_with_inchi("ZZZZZZZZZZZZZZ-ZZZZZZZZZZ-Z"), q)

    def test_stock_inchikeys_property_returns_set_like(self):
        q = InMemoryInchiKeyQuery(self.h5_path)
        keys = q.stock_inchikeys
        self.assertEqual(len(keys), 3)
        self.assertIn(KEY_A, keys)

    def test_price_raises_when_no_price_dict(self):
        q = InMemoryInchiKeyQuery(self.h5_path)
        with self.assertRaises(StockException):
            q.price(_make_mol_with_inchi(KEY_A))


class TestInMemoryInchiKeyQueryCSV(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmpdir, "stock.csv")
        df = pd.DataFrame({
            "inchi_key": [KEY_A, KEY_B, KEY_C],
            "price": [1.0, 2.5, 0.5],
        })
        df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_loads_from_csv_with_prices(self):
        q = InMemoryInchiKeyQuery(self.csv_path, price_col="price")
        self.assertEqual(len(q), 3)
        self.assertEqual(q.price(_make_mol_with_inchi(KEY_A)), 1.0)
        self.assertEqual(q.price(_make_mol_with_inchi(KEY_B)), 2.5)

    def test_loads_from_csv_without_prices(self):
        q = InMemoryInchiKeyQuery(self.csv_path)
        self.assertEqual(len(q), 3)


class TestInMemoryInchiKeyQueryText(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.txt_path = os.path.join(self.tmpdir, "stock.txt")
        with open(self.txt_path, "w") as fh:
            for k in [KEY_A, KEY_B, KEY_C]:
                fh.write(k + "\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_loads_from_text(self):
        q = InMemoryInchiKeyQuery(self.txt_path)
        self.assertEqual(len(q), 3)
        self.assertIn(_make_mol_with_inchi(KEY_A), q)


if __name__ == "__main__":
    unittest.main()
