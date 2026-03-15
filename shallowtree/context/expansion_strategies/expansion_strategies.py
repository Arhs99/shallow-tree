""" Module containing classes that implements different expansion policy strategies
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from shallowtree.utils.exceptions import PolicyException
from shallowtree.utils.logging import logger

if TYPE_CHECKING:
    from shallowtree.chem import TreeMolecule
    from shallowtree.chem.reaction import RetroReaction
    from shallowtree.context.config import Configuration
    from shallowtree.utils.type_utils import (
        List,
        Optional,
        Sequence,
        Tuple,
    )


class ExpansionStrategy(abc.ABC):
    """
    A base class for all expansion strategies.

    The strategy can be used by either calling the `get_actions` method
    of by calling the instantiated class with a list of molecule.

    .. code-block::

        expander = MyExpansionStrategy("dummy", config)
        actions, priors = expander.get_actions(molecules)
        actions, priors = expander(molecules)

    :param key: the key or label
    :param config: the configuration of the tree search
    """

    _required_kwargs: List[str] = []

    def __init__(self, key: str, config: Configuration, **kwargs: str):
        if any(name not in kwargs for name in self._required_kwargs):
            raise PolicyException(
                f"A {self.__class__.__name__} class needs to be initiated "
                f"with keyword arguments: {', '.join(self._required_kwargs)}"
            )
        self._config = config
        self._logger = logger()
        self.key = key

    def __call__(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        return self.get_actions(molecules, cache_molecules)

    @abc.abstractmethod
    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        Get all the probable actions of a set of molecules

        :param molecules: the molecules to consider
        :param cache_molecules: additional molecules to submit to the expansion
                                  policy but that only will be cached for later use
        :return: the actions and the priors of those actions
        """

    def reset_cache(self) -> None:
        """Reset the prediction cache"""


