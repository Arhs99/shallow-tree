"""
Tests for Redis cache functionality.

Uses fakeredis for mocking Redis without requiring a running Redis instance.
All tests are designed to run without external dependencies.
"""
import json
import pytest
from unittest.mock import MagicMock, patch

try:
    import fakeredis
    HAS_FAKEREDIS = True
except ImportError:
    HAS_FAKEREDIS = False

# Skip all tests if fakeredis not installed
pytestmark = pytest.mark.skipif(not HAS_FAKEREDIS, reason="fakeredis not installed")


class MockConfig:
    """Mock Configuration object for testing."""

    def __init__(self, model_path="/path/to/expansion.hdf5"):
        self.expansion_policy = MagicMock()
        self.expansion_policy._items = {
            "full": MagicMock(
                cutoff_number=50,
            )
        }
        self.filter_policy = MagicMock()
        self.filter_policy._items = {
            "all": MagicMock(filter_cutoff=0.05)
        }
        self.stock = MagicMock()
        self.stock._items = {"zinc": MagicMock()}


class MockTreeMolecule:
    """Mock TreeMolecule for testing."""

    def __init__(self, smiles: str):
        self.smiles = smiles
        self._inchi_key = f"INCHIKEY_{smiles.replace('=', '').replace('(', '').replace(')', '')}"

    @property
    def inchi_key(self):
        return self._inchi_key


def create_redis_cache_with_fakeredis(config, fake_redis_client):
    """Helper to create a RedisCache with a fakeredis backend."""
    from shallowtree.context.cache.redis_cache import RedisCache

    # Patch redis.Redis to return our fakeredis client
    with patch("redis.Redis", return_value=fake_redis_client):
        cache = RedisCache(config=config)
    return cache


@pytest.fixture
def fake_redis():
    """Create a fakeredis instance for testing."""
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def redis_cache(fake_redis):
    """Create a RedisCache instance with fakeredis backend."""
    config = MockConfig()
    return create_redis_cache_with_fakeredis(config, fake_redis)


class TestConfigHash:
    """Tests for config hash computation."""

    def test_config_hash_deterministic(self, redis_cache):
        """Same config should produce same hash."""
        hash1 = redis_cache._compute_config_hash()
        hash2 = redis_cache._compute_config_hash()
        assert hash1 == hash2

    def test_config_hash_changes_with_cutoff(self, fake_redis):
        """Different cutoff values should produce different hashes."""
        config1 = MockConfig()
        cache1 = create_redis_cache_with_fakeredis(config1, fake_redis)

        config2 = MockConfig()
        config2.expansion_policy._items["full"].cutoff_number = 100  # Different cutoff
        cache2 = create_redis_cache_with_fakeredis(config2, fake_redis)

        assert cache1._config_hash != cache2._config_hash

    def test_config_hash_length(self, redis_cache):
        """Hash should be 16 characters (truncated SHA256)."""
        assert len(redis_cache._config_hash) == 16


class TestCacheOperations:
    """Tests for cache get/set operations."""

    def test_cache_roundtrip(self, redis_cache):
        """Test set/get cache data."""
        inchi_key = "TESTINCHIKEY123"
        depth = 2
        score = 0.95

        # Set cache
        redis_cache.set_cache(inchi_key, depth, score)

        # Get cache
        result = redis_cache.get_cache(inchi_key)

        assert result is not None
        assert result[0] == depth
        assert result[1] == score

    def test_cache_miss(self, redis_cache):
        """Test cache miss returns None."""
        result = redis_cache.get_cache("NONEXISTENT_KEY")
        assert result is None

    def test_cache_overwrite(self, redis_cache):
        """Test that cache values can be overwritten."""
        inchi_key = "TESTINCHIKEY123"

        redis_cache.set_cache(inchi_key, 2, 0.5)
        redis_cache.set_cache(inchi_key, 1, 0.95)

        result = redis_cache.get_cache(inchi_key)
        assert result == (1, 0.95)


class TestSolvedOperations:
    """Tests for solved route get/set operations."""

    def test_solved_roundtrip(self, redis_cache):
        """Test set/get solved data with TreeMolecule reconstruction."""
        inchi_key = "TESTINCHIKEY123"
        reactants = [MockTreeMolecule("CCO"), MockTreeMolecule("CC(=O)O")]
        score = 0.95
        classification = "Ester hydrolysis"

        # Patch TreeMolecule to use our mock
        with patch(
            "shallowtree.context.cache.redis_cache.TreeMolecule",
            side_effect=lambda parent, smiles: MockTreeMolecule(smiles),
        ):
            redis_cache.set_solved(inchi_key, reactants, score, classification)
            result = redis_cache.get_solved(inchi_key)

        assert result is not None
        result_reactants, result_score, result_classification = result
        assert len(result_reactants) == 2
        assert result_reactants[0].smiles == "CCO"
        assert result_reactants[1].smiles == "CC(=O)O"
        assert result_score == score
        assert result_classification == classification

    def test_solved_miss(self, redis_cache):
        """Test solved miss returns None."""
        result = redis_cache.get_solved("NONEXISTENT_KEY")
        assert result is None


class TestBulkOperations:
    """Tests for bulk cache operations."""

    def test_get_cache_multi(self, redis_cache):
        """Test bulk cache retrieval."""
        # Set up some cache entries
        redis_cache.set_cache("KEY1", 1, 0.9)
        redis_cache.set_cache("KEY2", 2, 0.8)

        # Get multiple keys
        result = redis_cache.get_cache_multi(["KEY1", "KEY2", "KEY3"])

        assert len(result) == 3
        assert result["KEY1"] == (1, 0.9)
        assert result["KEY2"] == (2, 0.8)
        assert result["KEY3"] is None

    def test_get_cache_multi_empty(self, redis_cache):
        """Test bulk retrieval with empty list."""
        result = redis_cache.get_cache_multi([])
        assert result == {}


class TestKeyFormat:
    """Tests for Redis key formatting."""

    def test_key_format_cache(self, redis_cache):
        """Test cache key format."""
        key = redis_cache._make_key("cache", "TESTKEY123")
        assert key.startswith("shallowtree:")
        assert ":cache:TESTKEY123" in key
        assert redis_cache._config_hash in key

    def test_key_format_solved(self, redis_cache):
        """Test solved key format."""
        key = redis_cache._make_key("solved", "TESTKEY123")
        assert key.startswith("shallowtree:")
        assert ":solved:TESTKEY123" in key
        assert redis_cache._config_hash in key


class TestConnectionErrors:
    """Tests for connection error handling."""

    def test_fail_fast_on_connection_error(self):
        """Should raise CacheException if Redis unavailable."""
        import redis as redis_module
        from shallowtree.utils.exceptions import CacheException

        # Mock redis.Redis to raise ConnectionError on ping
        mock_client = MagicMock()
        mock_client.ping.side_effect = redis_module.ConnectionError("Connection refused")

        config = MockConfig()
        with patch("redis.Redis", return_value=mock_client):
            with pytest.raises(CacheException) as exc_info:
                from shallowtree.context.cache.redis_cache import RedisCache
                RedisCache(config=config)
            assert "Connection refused" in str(exc_info.value)


class TestIntegration:
    """Integration-style tests for cache behavior."""

    def test_config_isolation(self, fake_redis):
        """Different configs should have isolated cache namespaces."""
        config1 = MockConfig()
        cache1 = create_redis_cache_with_fakeredis(config1, fake_redis)

        config2 = MockConfig()
        config2.expansion_policy._items["full"].cutoff_number = 100  # Different config
        cache2 = create_redis_cache_with_fakeredis(config2, fake_redis)

        # Set value in cache1
        cache1.set_cache("SHARED_KEY", 1, 0.9)

        # Should not be visible in cache2 (different config hash)
        result = cache2.get_cache("SHARED_KEY")
        assert result is None

        # But visible in cache1
        result = cache1.get_cache("SHARED_KEY")
        assert result == (1, 0.9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
