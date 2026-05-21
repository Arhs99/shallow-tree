"""Tiny OrderedDict-backed LRU cache.

Intentionally minimal — only the surface needed by the TreeMolecule intern
cache: ``get``, ``__setitem__``, ``__contains__``, ``__len__``, ``clear``.
Kept here to avoid pulling in ``cachetools`` (removed during an earlier
LRU revert; see plans/lru_cache_attempt.md).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Bounded least-recently-used cache."""

    __slots__ = ("_maxsize", "_data")

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._maxsize = maxsize
        self._data: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return default

    def __setitem__(self, key: K, value: V) -> None:
        if key in self._data:
            self._data.move_to_end(key)
            self._data[key] = value
            return
        self._data[key] = value
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()
