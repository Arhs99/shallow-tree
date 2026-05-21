"""Tests for shallowtree.utils.lru.LRUCache."""

import unittest

from shallowtree.utils.lru import LRUCache


class TestLRUCache(unittest.TestCase):
    def test_basic_put_and_get(self):
        c = LRUCache(maxsize=3)
        c["a"] = 1
        c["b"] = 2
        self.assertEqual(c.get("a"), 1)
        self.assertEqual(c.get("b"), 2)
        self.assertIsNone(c.get("missing"))
        self.assertEqual(c.get("missing", "default"), "default")

    def test_eviction_least_recently_used(self):
        c = LRUCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3  # evicts "a"
        self.assertNotIn("a", c)
        self.assertIn("b", c)
        self.assertIn("c", c)

    def test_access_promotes_to_mru(self):
        c = LRUCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        _ = c.get("a")  # promotes "a"
        c["c"] = 3  # evicts "b" since "a" is MRU
        self.assertIn("a", c)
        self.assertNotIn("b", c)
        self.assertIn("c", c)

    def test_overwrite_keeps_size(self):
        c = LRUCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        c["a"] = 99
        self.assertEqual(len(c), 2)
        self.assertEqual(c.get("a"), 99)

    def test_clear(self):
        c = LRUCache(maxsize=10)
        c["a"] = 1
        c["b"] = 2
        c.clear()
        self.assertEqual(len(c), 0)
        self.assertNotIn("a", c)

    def test_zero_maxsize_rejected(self):
        with self.assertRaises(ValueError):
            LRUCache(maxsize=0)


if __name__ == "__main__":
    unittest.main()
