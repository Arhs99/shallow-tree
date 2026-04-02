"""Tests for shallowtree.context.config — Configuration loading."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tempfile
import unittest

import yaml

from shallowtree.context.config import Configuration


class TestConfigFromFile(unittest.TestCase):
    """Test Configuration.from_file with env var substitution."""

    def test_env_var_substitution(self):
        yaml_str = "test_path: ${TEST_SHALLOW_TREE_PATH}\nexpansion: {}\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_str)
            f.flush()
            fname = f.name

        try:
            os.environ["TEST_SHALLOW_TREE_PATH"] = "/tmp/test"
            result = Configuration.from_file(fname)
            self.assertEqual(result["test_path"], "/tmp/test")
        finally:
            os.unlink(fname)
            del os.environ["TEST_SHALLOW_TREE_PATH"]

    def test_missing_env_var_raises_value_error(self):
        yaml_str = "path: ${NONEXISTENT_SHALLOW_TREE_VAR}\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_str)
            f.flush()
            fname = f.name

        try:
            os.environ.pop("NONEXISTENT_SHALLOW_TREE_VAR", None)
            with self.assertRaises(ValueError):
                Configuration.from_file(fname)
        finally:
            os.unlink(fname)


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
