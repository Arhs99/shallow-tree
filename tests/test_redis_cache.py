"""
Tests for Redis cache functionality.

Uses fakeredis for mocking Redis without requiring a running Redis instance.
"""
import json
import unittest
from unittest.mock import MagicMock, patch

try:
    import fakeredis
    HAS_FAKEREDIS = True
except ImportError:
    HAS_FAKEREDIS = False


class MockTreeMolecule:
    """Mock TreeMolecule for testing."""

    def __init__(self, smiles: str):
        self.smiles = smiles
        self._inchi_key = f"INCHIKEY_{smiles.replace('=', '').replace('(', '').replace(')', '')}"

    @property
    def inchi_key(self):
        return self._inchi_key


def _mock_policies():
    """Create mock filter_policy, expansion_policy, stock."""
    expansion_policy = MagicMock()
    expansion_policy._items = {"full": MagicMock(cutoff_number=50)}
    filter_policy = MagicMock()
    filter_policy._items = {"all": MagicMock(filter_cutoff=0.05)}
    stock = MagicMock()
    stock._items = {"zinc": MagicMock()}
    return filter_policy, expansion_policy, stock


def _create_cache(fake_redis_client, filter_policy=None, expansion_policy=None, stock=None):
    """Create a RedisCache with a fakeredis backend."""
    from shallowtree.context.cache.redis_cache import RedisCache

    if filter_policy is None:
        filter_policy, expansion_policy, stock = _mock_policies()

    with patch("redis.Redis", return_value=fake_redis_client):
        cache = RedisCache(
            filter_policy=filter_policy,
            expansion_policy=expansion_policy,
            stock=stock,
        )
    return cache


@unittest.skipUnless(HAS_FAKEREDIS, "fakeredis not installed")
class TestConfigHash(unittest.TestCase):

    def setUp(self):
        self.fake_redis = fakeredis.FakeRedis(decode_responses=True)
        self.cache = _create_cache(self.fake_redis)

    def test_config_hash_deterministic(self):
        fp, ep, st = _mock_policies()
        hash1 = self.cache._compute_config_hash(fp, ep, st)
        hash2 = self.cache._compute_config_hash(fp, ep, st)
        self.assertEqual(hash1, hash2)

    def test_config_hash_changes_with_cutoff(self):
        fp1, ep1, st1 = _mock_policies()
        cache1 = _create_cache(self.fake_redis, fp1, ep1, st1)

        fp2, ep2, st2 = _mock_policies()
        ep2._items["full"].cutoff_number = 100
        cache2 = _create_cache(self.fake_redis, fp2, ep2, st2)

        self.assertNotEqual(cache1._config_hash, cache2._config_hash)

    def test_config_hash_length(self):
        self.assertEqual(len(self.cache._config_hash), 16)


@unittest.skipUnless(HAS_FAKEREDIS, "fakeredis not installed")
class TestCacheOperations(unittest.TestCase):

    def setUp(self):
        self.fake_redis = fakeredis.FakeRedis(decode_responses=True)
        self.cache = _create_cache(self.fake_redis)

    def test_cache_roundtrip(self):
        self.cache.set_cache("TESTKEY", 2, 0.95)
        result = self.cache.get_cache("TESTKEY")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 2)
        self.assertEqual(result[1], 0.95)

    def test_cache_miss(self):
        result = self.cache.get_cache("NONEXISTENT")
        self.assertIsNone(result)

    def test_cache_overwrite(self):
        self.cache.set_cache("TESTKEY", 2, 0.5)
        self.cache.set_cache("TESTKEY", 1, 0.95)
        result = self.cache.get_cache("TESTKEY")
        self.assertEqual(result, (1, 0.95))


@unittest.skipUnless(HAS_FAKEREDIS, "fakeredis not installed")
class TestSolvedOperations(unittest.TestCase):

    def setUp(self):
        self.fake_redis = fakeredis.FakeRedis(decode_responses=True)
        self.cache = _create_cache(self.fake_redis)

    def test_solved_roundtrip(self):
        reactants = [MockTreeMolecule("CCO"), MockTreeMolecule("CC(=O)O")]
        with patch(
            "shallowtree.context.cache.redis_cache.TreeMolecule",
            side_effect=lambda parent, smiles: MockTreeMolecule(smiles),
        ):
            self.cache.set_solved("TESTKEY", reactants, 0.95, "Ester hydrolysis")
            result = self.cache.get_solved("TESTKEY")

        self.assertIsNotNone(result)
        r_reactants, r_score, r_class = result
        self.assertEqual(len(r_reactants), 2)
        self.assertEqual(r_reactants[0].smiles, "CCO")
        self.assertEqual(r_score, 0.95)
        self.assertEqual(r_class, "Ester hydrolysis")

    def test_solved_miss(self):
        self.assertIsNone(self.cache.get_solved("NONEXISTENT"))


@unittest.skipUnless(HAS_FAKEREDIS, "fakeredis not installed")
class TestBulkOperations(unittest.TestCase):

    def setUp(self):
        self.fake_redis = fakeredis.FakeRedis(decode_responses=True)
        self.cache = _create_cache(self.fake_redis)

    def test_get_cache_multi(self):
        self.cache.set_cache("KEY1", 1, 0.9)
        self.cache.set_cache("KEY2", 2, 0.8)
        result = self.cache.get_cache_multi(["KEY1", "KEY2", "KEY3"])
        self.assertEqual(result["KEY1"], (1, 0.9))
        self.assertEqual(result["KEY2"], (2, 0.8))
        self.assertIsNone(result["KEY3"])

    def test_get_cache_multi_empty(self):
        self.assertEqual(self.cache.get_cache_multi([]), {})


@unittest.skipUnless(HAS_FAKEREDIS, "fakeredis not installed")
class TestKeyFormat(unittest.TestCase):

    def setUp(self):
        self.fake_redis = fakeredis.FakeRedis(decode_responses=True)
        self.cache = _create_cache(self.fake_redis)

    def test_key_format_cache(self):
        key = self.cache._make_key("cache", "TESTKEY123")
        self.assertTrue(key.startswith("shallowtree:"))
        self.assertIn(":cache:TESTKEY123", key)

    def test_key_format_solved(self):
        key = self.cache._make_key("solved", "TESTKEY123")
        self.assertTrue(key.startswith("shallowtree:"))
        self.assertIn(":solved:TESTKEY123", key)


@unittest.skipUnless(HAS_FAKEREDIS, "fakeredis not installed")
class TestConnectionErrors(unittest.TestCase):

    def test_fail_fast_on_connection_error(self):
        import redis as redis_module
        from shallowtree.utils.exceptions import CacheException
        from shallowtree.context.cache.redis_cache import RedisCache

        mock_client = MagicMock()
        mock_client.ping.side_effect = redis_module.ConnectionError("Connection refused")

        fp, ep, st = _mock_policies()
        with patch("redis.Redis", return_value=mock_client):
            with self.assertRaises(CacheException):
                RedisCache(filter_policy=fp, expansion_policy=ep, stock=st)


@unittest.skipUnless(HAS_FAKEREDIS, "fakeredis not installed")
class TestConfigIsolation(unittest.TestCase):

    def test_config_isolation(self):
        fake_redis = fakeredis.FakeRedis(decode_responses=True)

        fp1, ep1, st1 = _mock_policies()
        cache1 = _create_cache(fake_redis, fp1, ep1, st1)

        fp2, ep2, st2 = _mock_policies()
        ep2._items["full"].cutoff_number = 100
        cache2 = _create_cache(fake_redis, fp2, ep2, st2)

        cache1.set_cache("SHARED_KEY", 1, 0.9)
        self.assertIsNone(cache2.get_cache("SHARED_KEY"))
        self.assertEqual(cache1.get_cache("SHARED_KEY"), (1, 0.9))


if __name__ == "__main__":
    unittest.main()
