"""Tests for shallowtree.context.config — Configuration loading."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tempfile
import unittest

from shallowtree.context.config import Configuration


class TestConfigEquality(unittest.TestCase):
    """Test Configuration __eq__."""

    def test_equal_configs(self):
        c1 = Configuration()
        c2 = Configuration()
        self.assertEqual(c1, c2)

    def test_not_equal_to_non_config(self):
        c1 = Configuration()
        self.assertNotEqual(c1, "not a config")
        self.assertNotEqual(c1, 42)


if __name__ == "__main__":
    unittest.main()
