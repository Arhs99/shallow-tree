"""Redis-based persistent cache for retrosynthetic search results."""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING

from shallowtree.chem import TreeMolecule
from shallowtree.utils.exceptions import CacheException

if TYPE_CHECKING:
    from shallowtree.context.config import Configuration
    from shallowtree.utils.type_utils import Dict, List, Optional, Tuple


class RedisCache:
    """Persistent Redis cache for sharing search results across processes.

    Stores both branch pruning data (depth, score) and route reconstruction
    data (reactants, score, classification) with automatic config-based
    namespace isolation.
    """

    def __init__(
        self,
        config: "Configuration",
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: float = 5.0,
    ) -> None:
        """Initialize Redis connection with fail-fast behavior.

        Args:
            config: Configuration object for computing config hash.
            host: Redis server hostname.
            port: Redis server port.
            db: Redis database number.
            password: Optional Redis password.
            socket_timeout: Connection timeout in seconds.

        Raises:
            CacheException: If Redis connection fails.
        """
        try:
            import redis
        except ImportError:
            raise CacheException(
                "redis package not installed. Install with: poetry install -E cache"
            )

        self._config = config
        self._config_hash = self._compute_config_hash()

        try:
            self._client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                socket_timeout=socket_timeout,
                decode_responses=True,
            )
            # Test connection immediately (fail-fast)
            self._client.ping()
        except redis.ConnectionError as e:
            raise CacheException(f"Failed to connect to Redis at {host}:{port}: {e}")
        except redis.AuthenticationError as e:
            raise CacheException(f"Redis authentication failed: {e}")

    def _compute_config_hash(self) -> str:
        """Compute deterministic hash of config fields that affect search results.

        Uses policy/stock key names and relevant settings to create a unique
        identifier for this configuration. Different configs get separate
        cache namespaces.
        """
        hash_data = {}

        # Expansion policy - use key names and cutoff settings
        for name, strategy in self._config.expansion_policy._items.items():
            hash_data[f"expansion.{name}"] = name
            hash_data[f"expansion.{name}.cutoff"] = getattr(strategy, "cutoff_number", 50)

        # Filter policy - use key names and filter cutoff
        for name, strategy in self._config.filter_policy._items.items():
            hash_data[f"filter.{name}"] = name
            hash_data[f"filter.{name}.cutoff"] = getattr(strategy, "filter_cutoff", 0.05)

        # Stock - use key names
        for name in self._config.stock._items.keys():
            hash_data[f"stock.{name}"] = name

        # Create deterministic hash
        json_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def _make_key(self, key_type: str, inchi_key: str) -> str:
        """Create Redis key with namespace and config hash prefix.

        Args:
            key_type: Either 'cache' or 'solved'.
            inchi_key: The molecule's InChI key.

        Returns:
            Formatted Redis key string.
        """
        return f"shallowtree:{self._config_hash}:{key_type}:{inchi_key}"

    def get_cache(self, inchi_key: str) -> Optional[Tuple[int, float]]:
        """Get cached depth and score for a molecule.

        Args:
            inchi_key: The molecule's InChI key.

        Returns:
            Tuple of (depth, score) if found, None otherwise.
        """
        key = self._make_key("cache", inchi_key)
        data = self._client.get(key)
        if data is None:
            return None
        parsed = json.loads(data)
        return (parsed["depth"], parsed["score"])

    def set_cache(self, inchi_key: str, depth: int, score: float) -> None:
        """Store depth and score for a molecule.

        Args:
            inchi_key: The molecule's InChI key.
            depth: Search depth at which this result was computed.
            score: Synthesis feasibility score.
        """
        key = self._make_key("cache", inchi_key)
        data = json.dumps({"depth": depth, "score": score, "ts": int(time.time())})
        self._client.set(key, data)

    def get_solved(
        self, inchi_key: str
    ) -> Optional[Tuple[List[TreeMolecule], float, str]]:
        """Get solved route data for a molecule.

        Args:
            inchi_key: The molecule's InChI key.

        Returns:
            Tuple of (reactants, score, classification) if found, None otherwise.
            Reactants are reconstructed as TreeMolecule objects.
        """
        key = self._make_key("solved", inchi_key)
        data = self._client.get(key)
        if data is None:
            return None
        parsed = json.loads(data)
        # Reconstruct TreeMolecule objects from SMILES
        reactants = [
            TreeMolecule(parent=None, smiles=smi)
            for smi in parsed["reactants_smiles"]
        ]
        return (reactants, parsed["score"], parsed["classification"])

    def set_solved(
        self,
        inchi_key: str,
        reactants: List[TreeMolecule],
        score: float,
        classification: str,
    ) -> None:
        """Store solved route data for a molecule.

        Args:
            inchi_key: The molecule's InChI key.
            reactants: List of reactant TreeMolecule objects.
            score: Synthesis feasibility score.
            classification: Reaction classification string.
        """
        key = self._make_key("solved", inchi_key)
        data = json.dumps({
            "reactants_smiles": [mol.smiles for mol in reactants],
            "score": score,
            "classification": classification,
            "ts": int(time.time()),
        })
        self._client.set(key, data)

    def get_cache_multi(
        self, inchi_keys: List[str]
    ) -> Dict[str, Optional[Tuple[int, float]]]:
        """Get cached data for multiple molecules in one round-trip.

        Args:
            inchi_keys: List of InChI keys to look up.

        Returns:
            Dictionary mapping inchi_key to (depth, score) or None.
        """
        if not inchi_keys:
            return {}

        keys = [self._make_key("cache", ik) for ik in inchi_keys]
        values = self._client.mget(keys)

        result = {}
        for inchi_key, data in zip(inchi_keys, values):
            if data is None:
                result[inchi_key] = None
            else:
                parsed = json.loads(data)
                result[inchi_key] = (parsed["depth"], parsed["score"])
        return result
