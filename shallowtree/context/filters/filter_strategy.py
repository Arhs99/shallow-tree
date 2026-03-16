""" Module containing classes that implements different filter policy strategies
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from shallowtree.utils.exceptions import (
    PolicyException,
)
from shallowtree.utils.logging import logger

if TYPE_CHECKING:
    from shallowtree.chem.reaction import RetroReaction
    from shallowtree.context.config import Configuration
    from shallowtree.utils.type_utils import Any, List


class FilterStrategy(abc.ABC):
    """
    A base class for all filter strategies.

    The filter can be applied by either calling the `apply` method
    of by calling the instantiated class with a reaction.

    .. code-block::

        filter = MyFilterStrategy("dummy", config)
        filter.apply(reaction)
        filter(reaction)

    :param key: the key or label
    :param config: the configuration of the tree search
    """

    _required_kwargs: List[str] = []

    def __init__(self, key: str, config: Configuration, **kwargs: Any) -> None:
        if any(name not in kwargs for name in self._required_kwargs):
            raise PolicyException(
                f"A {self.__class__.__name__} class needs to be initiated "
                f"with keyword arguments: {', '.join(self._required_kwargs)}"
            )
        self._config = config
        self._logger = logger()
        self.key = key

    def __call__(self, reaction: RetroReaction) -> None:
        self.apply(reaction)

    @abc.abstractmethod
    def apply(self, reaction: RetroReaction) -> None:
        """
        Apply the filter on the reaction. If the reaction
        should be rejected a `RejectionException` is raised

        :param reaction: the reaction to filter
        :raises: if the reaction should be rejected.
        """


FILTER_STRATEGY_ALIAS = {
    "feasibility": "QuickKerasFilter",
    "quick_keras_filter": "QuickKerasFilter",
    "reactants_count": "ReactantsCountFilter",
}
