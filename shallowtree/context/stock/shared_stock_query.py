"""Stock query adapter over a SharedInchiKeySet.

Drop-in substitute for :class:`InMemoryInchiKeyQuery` in workers that need
to delegate membership checks to a shared-memory-backed key set built by
the parent process.
"""

from __future__ import annotations

from shallowtree.chem.mol import Molecule
from shallowtree.context.stock.queries import StockQueryMixin
from shallowtree.context.stock.shared_inchi_key_set import SharedInchiKeySet


class SharedInchiKeyQuery(StockQueryMixin):
    """StockQueryMixin-compatible wrapper around a SharedInchiKeySet."""

    def __init__(self, shared_set: SharedInchiKeySet) -> None:
        self._set = shared_set

    def __contains__(self, mol: Molecule) -> bool:
        return mol.inchi_key in self._set

    def __len__(self) -> int:
        return len(self._set)
