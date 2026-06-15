"""Redis-based persistent cache for retrosynthetic search results."""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING

from shallowtree.configs.cache_configuration import CacheConfiguration
from shallowtree.context.cache.redis_data_dto import RedisDataDTO
from shallowtree.context.cache.redis_resolved_data_dto import RedisResolvedDataDTO

from shallowtree.context.policy.filter_policy import FilterPolicy

from shallowtree.context.policy.expansion_policy import ExpansionPolicy

from shallowtree.chem.molecules.tree_molecule import TreeMolecule
from shallowtree.context.stock.stock import Stock
from shallowtree.utils.exceptions import CacheException

if TYPE_CHECKING:
    from shallowtree.utils.type_utils import Dict, List, Optional, Tuple


class RedisCache:
    """Persistent Redis cache for sharing search results across processes.

    Stores both branch pruning data (depth, score) and route reconstruction
    data (reactants, score, classification) with automatic config-based
    namespace isolation.
    """

    def __init__(self, filter_policy: FilterPolicy, expansion_policy: ExpansionPolicy, stock: Stock,
                 cache_config: CacheConfiguration) -> None:
        """Initialize Redis connection with fail-fast behavior.
        Raises:
            CacheException: If Redis connection fails.
        """
        try:
            import redis
        except ImportError:
            raise CacheException(
                "redis package not installed. Install with: poetry install -E cache"
            )

        self._config_hash = self._compute_config_hash(filter_policy, expansion_policy, stock)
        self._namespace = cache_config.namespace

        try:
            self._client = redis.Redis(
                host=cache_config.host,
                port=cache_config.port,
                db=cache_config.db,
                password=cache_config.password,
                socket_timeout=cache_config.socket_timeout,
                decode_responses=True,
            )
            # Test connection immediately (fail-fast)
            self._client.ping()
        except redis.ConnectionError as e:
            raise CacheException(f"Failed to connect to Redis at {cache_config.host}:{cache_config.port}: {e}")
        except redis.AuthenticationError as e:
            raise CacheException(f"Redis authentication failed: {e}")

    def _compute_config_hash(self, filter_policy: FilterPolicy, expansion_policy: ExpansionPolicy, stock) -> str:
        """Compute deterministic hash of config fields that affect search results.

        Uses policy/stock key names and relevant settings to create a unique
        identifier for this configuration. Different configs get separate
        cache namespaces.
        """
        hash_data = {}

        # Expansion policy - use key names and cutoff settings
        for name in expansion_policy.items:
            hash_data[f"expansion.{name}"] = name
            strategy = expansion_policy.get_item(name)
            hash_data[f"expansion.{name}.cutoff"] = strategy.cutoff_number

        # Filter policy - use key names and filter cutoff
        for name in filter_policy.items:
            hash_data[f"filter.{name}"] = name
            strategy = filter_policy.get_item(name)
            hash_data[f"filter.{name}.cutoff"] = strategy.filter_cutoff if strategy.filter_cutoff else 0.05

        # Stock - use key names
        for name in stock.items:
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
        return f"shallowtree:{self._config_hash}:{self._namespace}:{key_type}:{inchi_key}"

    def get_cache(self, inchi_key: str) -> RedisDataDTO:
        """Get cached depth, score and resolved flag for a molecule.

        Args:
            inchi_key: The molecule's InChI key.

        Returns:
            Tuple of (depth, score, resolved) if found, None otherwise.
            ``resolved`` defaults to False for legacy entries written before
            the resolution gate existed.
        """
        key = self._make_key("cache", inchi_key)
        data = self._client.get(key)
        if data is None:
            return RedisDataDTO(inchi_key=inchi_key, exists=False)
        parsed = json.loads(data)
        dto = RedisDataDTO(inchi_key=inchi_key, exists=True, **parsed)
        return dto

    def set_cache(self, inchi_key: str, depth: int, score: float, resolved: bool = False) -> None:
        """Store depth, score and resolved flag for a molecule.

        Args:
            inchi_key: The molecule's InChI key.
            depth: Search depth at which this result was computed.
            score: Synthesis feasibility score.
            resolved: Whether the route bottoms out entirely in stock.
        """
        key = self._make_key("cache", inchi_key)
        data = json.dumps({"depth": depth, "score": score, "resolved": resolved, "timestamp": int(time.time())})
        self._client.set(key, data)

    def get_solved(self, inchi_key: str) -> RedisResolvedDataDTO:
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
            return RedisResolvedDataDTO(inchi_key=inchi_key, exists=False)
        parsed = json.loads(data)
        # Reconstruct TreeMolecule objects from SMILES
        reactants = [
            TreeMolecule(parent=None, smiles=smi)
            for smi in parsed["reactants_smiles"]
        ]
        return RedisResolvedDataDTO(inchi_key=inchi_key, reactants=reactants, **parsed, exists=True)

    def set_solved(self, inchi_key: str, reactants: List[TreeMolecule], score: float, classification: str,
                   start_time: float) -> None:
        """Store solved route data for a molecule.

        Args:
            inchi_key: The molecule's InChI key.
            reactants: List of reactant TreeMolecule objects.
            score: Synthesis feasibility score.
            classification: Reaction classification string.
            start_time: Wall-clock time when the search for this root began; used
                to record how long solving took.
        """
        key = self._make_key("solved", inchi_key)
        data = json.dumps({
            "reactants_smiles": [mol.smiles for mol in reactants],
            "score": score,
            "classification": classification,
            "timestamp": int(time.time()),
            "duration_seconds": int(time.time() - start_time),
        })
        self._client.set(key, data)
