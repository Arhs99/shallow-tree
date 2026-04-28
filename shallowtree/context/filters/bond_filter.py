from __future__ import annotations

from typing import Sequence, List

from shallowtree.chem import RetroReaction
from shallowtree.configs.filter_configuration import FilterConfiguration
from shallowtree.context.filters.filter_strategy import FilterStrategy
from shallowtree.utils.bonds import BrokenBonds
from shallowtree.utils.exceptions import RejectionException


class BondFilter(FilterStrategy):
    """
    Check if focussed bonds to freeze stay frozen in a reaction.

    :param key: the key or label
    :param config: the configuration of the tree search
    """
    _required_kwargs: List[str] = ["freeze_bonds"]

    def __init__(self, key: str, config: FilterConfiguration) -> None:
        super().__init__(key)

        self._freeze_bonds: Sequence[Sequence[int]] = config.freeze_bonds # TODO: figure how to use this
        self._broken_bonds = BrokenBonds(self._freeze_bonds)
        self._logger.info(
            f"Loading bond filter to {key} with {len(self._freeze_bonds)} "
            "bonds to freeze"
        )

    def apply(self, reaction: RetroReaction) -> None:
        broken_frozen_bonds = self._broken_bonds(reaction)
        if len(broken_frozen_bonds) > 0:
            raise RejectionException(
                f"{reaction} was filtered out as the focussed bonds "
                f"'{broken_frozen_bonds}' were found to be broken in the reaction"
            )
