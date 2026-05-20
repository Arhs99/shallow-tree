"""Compact, sorted, fixed-width container for InChI keys.

Stores N keys as a single contiguous numpy ``|S27`` array (27 bytes per
record). ``__contains__`` uses ``np.searchsorted`` for O(log N) lookup.
Resident memory is ~27 * N bytes plus a small numpy header, compared to
~115 bytes per entry for a ``frozenset[str]``.

This primitive deliberately exposes its underlying buffer via
``raw_buffer()`` so a sibling shared-memory variant can construct the
same logical set from a ``SharedMemory`` block without an extra copy.
"""

from __future__ import annotations

from typing import Iterable, Union

import numpy as np

INCHI_KEY_LEN = 27


class PackedInchiKeySet:
    """Sorted, deduplicated, fixed-width packed set of 27-byte InChI keys."""

    __slots__ = ("_array",)

    def __init__(self, sorted_unique_array: np.ndarray) -> None:
        if sorted_unique_array.dtype != np.dtype(f"|S{INCHI_KEY_LEN}"):
            raise ValueError(
                f"expected dtype |S{INCHI_KEY_LEN}, got {sorted_unique_array.dtype}"
            )
        self._array = sorted_unique_array

    @classmethod
    def from_iterable(cls, keys: Iterable[Union[str, bytes]]) -> "PackedInchiKeySet":
        """Build from any iterable of InChI keys (str or bytes).

        Non-27-char entries are silently dropped; this mirrors the existing
        ``frozenset`` behaviour where the caller is responsible for clean input.
        """
        packed = np.fromiter(
            (cls._encode(k) for k in keys if cls._is_valid(k)),
            dtype=f"|S{INCHI_KEY_LEN}",
        )
        unique_sorted = np.unique(packed)
        return cls(unique_sorted)

    @staticmethod
    def _encode(key: Union[str, bytes]) -> bytes:
        return key.encode("ascii") if isinstance(key, str) else key

    @staticmethod
    def _is_valid(key: Union[str, bytes]) -> bool:
        if isinstance(key, str):
            return len(key) == INCHI_KEY_LEN and key.isascii()
        if isinstance(key, (bytes, bytearray)):
            return len(key) == INCHI_KEY_LEN
        return False

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            try:
                kb = key.encode("ascii")
            except UnicodeEncodeError:
                return False
        elif isinstance(key, (bytes, bytearray)):
            kb = bytes(key)
        else:
            return False
        if len(kb) != INCHI_KEY_LEN:
            return False
        idx = int(np.searchsorted(self._array, kb))
        return idx < self._array.size and self._array[idx] == kb

    def __len__(self) -> int:
        return int(self._array.size)

    def __iter__(self):
        for raw in self._array:
            yield raw.decode("ascii")

    def raw_buffer(self) -> memoryview:
        """Return the underlying packed bytes (read-only view).

        Length is ``len(self) * INCHI_KEY_LEN``. Used by the shared-memory
        variant to copy into a ``SharedMemory`` block.
        """
        return memoryview(self._array).cast("B")
