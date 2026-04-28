from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Tuple, List

from shallowtree.context.collection import ContextCollection
from shallowtree.context.expansion_strategies.expansion_strategies import ExpansionStrategy
from shallowtree.utils.exceptions import PolicyException

if TYPE_CHECKING:
    from shallowtree.chem import TreeMolecule
    from shallowtree.chem.reaction import RetroReaction


class ExpansionPolicy(ContextCollection):
    """
    An abstraction of an expansion policy.

    This policy provides actions that can be applied to a molecule

    :param config: the configuration of the tree search
    """

    _collection_name = "expansion policy"

    def __init__(self):
        super().__init__()

    def get_actions(self, molecules: Sequence[TreeMolecule], cache_molecules: Sequence[TreeMolecule] = None, ) \
            -> Tuple[List[RetroReaction], List[float]]:
        """
        Get all the probable actions of a set of molecules, using the selected policies

        :param molecules: the molecules to consider
        :param cache_molecules: additional molecules that potentially are sent to
                                  the expansion model but for which predictions are not returned
        :return: the actions and the priors of those actions
        :raises: PolicyException: if the policy isn't selected
        """
        if not self.selection:
            raise PolicyException("No expansion policy selected")

        all_possible_actions = []
        all_priors = []
        for name in self.selection:
            possible_actions, priors = self[name].get_actions(molecules, cache_molecules)
            all_possible_actions.extend(possible_actions)
            all_priors.extend(priors)
        return all_possible_actions, all_priors

    def load(self, source: ExpansionStrategy) -> None:  # type: ignore
        """
        Add a pre-initialized expansion strategy object to the policy

        :param source: the item to add
        """
        if not isinstance(source, ExpansionStrategy):
            raise PolicyException("Only objects of classes inherited from ExpansionStrategy can be added")
        self._items[source.key] = source

    def reset_cache(self) -> None:
        """
        Reset the cache on all loaded policies
        """
        for policy in self._items.values():
            policy.reset_cache()


