"""Cross-process compact InChI-key set backed by ``multiprocessing.shared_memory``.

A parent process calls :meth:`SharedInchiKeySet.build` once to create a
SharedMemory block holding sorted, deduplicated, fixed-width 27-byte InChI
keys. Worker processes call :meth:`SharedInchiKeySet.attach` with the block
name to get a read-only view. Lookups go through binary search on the
shared bytes (no per-call Python-level allocation of the underlying data).

Lifecycle:

    parent: shared = SharedInchiKeySet.build(keys)
            ...launch workers passing shared.shm_name, len(shared)...
            shared.unlink()   # parent unlinks at shutdown

    worker: shared = SharedInchiKeySet.attach(shm_name, count)
            ...use shared in __contains__ checks...
            shared.close()    # workers only close, never unlink

``unlink()`` is owner-only and raises on workers; ``close()`` is safe in
either role.
"""

from __future__ import annotations

from multiprocessing import shared_memory
from typing import Iterable, Union

import numpy as np

from shallowtree.context.stock.packed_inchi_key_set import (
    INCHI_KEY_LEN,
    PackedInchiKeySet,
)


class SharedInchiKeySet:
    """Read-only set-like view over a SharedMemory block of 27-byte InChI keys."""

    __slots__ = ("_shm", "_count", "_owner", "_inner")

    def __init__(self, shm: shared_memory.SharedMemory, count: int, owner: bool) -> None:
        self._shm = shm
        self._count = count
        self._owner = owner
        # Numpy view directly into the shared buffer. Workers get a read-only
        # view because np.frombuffer respects the underlying memoryview's
        # writability and we explicitly make worker buffers read-only at
        # attach time.
        arr = np.frombuffer(self._shm.buf, dtype=f"|S{INCHI_KEY_LEN}", count=count)
        self._inner = PackedInchiKeySet(arr)

    @classmethod
    def build(cls, keys: Iterable[Union[str, bytes]]) -> "SharedInchiKeySet":
        """Create + populate a new SharedMemory block from an iterable of keys."""
        # Build sorted/dedup packed array using the in-process primitive so
        # encoding + validation rules stay in one place.
        prepared = PackedInchiKeySet.from_iterable(keys)
        packed_bytes = bytes(prepared.raw_buffer())
        size = len(packed_bytes)
        if size == 0:
            raise ValueError("cannot build SharedInchiKeySet from empty key set")
        shm = shared_memory.SharedMemory(create=True, size=size)
        shm.buf[:size] = packed_bytes
        return cls(shm, count=len(prepared), owner=True)

    @classmethod
    def attach(cls, shm_name: str, count: int) -> "SharedInchiKeySet":
        """Attach to an existing SharedMemory block by name."""
        shm = shared_memory.SharedMemory(name=shm_name)
        return cls(shm, count=count, owner=False)

    @property
    def shm_name(self) -> str:
        return self._shm.name

    def __contains__(self, key: object) -> bool:
        return key in self._inner

    def __len__(self) -> int:
        return self._count

    def close(self) -> None:
        """Detach from the SharedMemory block (safe for owner or worker)."""
        # Drop the numpy view first; numpy holds a memoryview into shm.buf
        # which can block close on some platforms.
        self._inner = None  # type: ignore[assignment]
        self._shm.close()

    def unlink(self) -> None:
        """Remove the SharedMemory block (owner only)."""
        if not self._owner:
            raise RuntimeError("only the owning process may unlink a SharedInchiKeySet")
        self.close()
        self._shm.unlink()
