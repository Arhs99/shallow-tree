from __future__ import annotations

from shallowtree.chem import RetroReaction
from shallowtree.context.collection import ContextCollection
from shallowtree.context.filters.filter_strategy import FilterStrategy
from shallowtree.utils.exceptions import PolicyException


class FilterPolicy(ContextCollection):
    """
    An abstraction of a filter policy.

    This policy provides a query on a reaction to determine whether it should be rejected

    :param config: the configuration of the tree search
    """

    _collection_name = "filter policy"

    def __init__(self,  source: FilterStrategy):
        super().__init__()
        self.load(source)

    def __call__(self, reaction: RetroReaction):
        return self.apply(reaction)

    def apply(self, reaction: RetroReaction) -> None:
        """
        Apply the all the selected filters on the reaction. If the reaction
        should be rejected a `RejectionException` is raised

        :param reaction: the reaction to filter
        :raises: if the reaction should be rejected or if a policy is selected
        """
        if not self.selection:
            raise PolicyException("No filter policy selected")

        for name in self.selection:
            self[name](reaction)

    def load(self, source: FilterStrategy):
        """
        Add a pre-initialized filter strategy object to the policy

        :param source: the item to add
        """
        if not isinstance(source, FilterStrategy):
            raise PolicyException(
                "Only objects of classes inherited from FilterStrategy can be added"
            )
        self._items[source.key] = source

    def reset_cache(self):
        """Reset filtering cache."""
        if not self.selection:
            return

        for name in self.selection:
            if hasattr(self[name], "reset_cache"):
                self[name].reset_cache()
