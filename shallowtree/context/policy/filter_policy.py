from __future__ import annotations

from typing import Any

from shallowtree.chem import RetroReaction
from shallowtree.context.collection import ContextCollection
from shallowtree.context.filters.filter_strategy import FilterStrategy
from shallowtree.context.filters.quick_keras_filter import QuickKerasFilter
from shallowtree.context.filters.filter_strategy import FILTER_STRATEGY_ALIAS

from shallowtree.utils.exceptions import PolicyException
from shallowtree.utils.loading import load_dynamic_class
from shallowtree.context.filters.filter_strategy import (
    __name__ as filter_strategy_module,
)


class FilterPolicy(ContextCollection):
    """
    An abstraction of a filter policy.

    This policy provides a query on a reaction to determine whether it should be rejected

    :param config: the configuration of the tree search
    """

    _collection_name = "filter policy"

    def __init__(self,  source: FilterStrategy):
        super().__init__()
        # self._config = config
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

    def load_from_config(self, **config: Any):
        """
        Load one or more filter policy from a configuration

        The format should be
        key:
            type: name of the filter class or custom_package.custom_model.CustomClass
            model: path_to_model
            other settings or params
        or
        key: path_to_model

        :param config: the configuration
        """
        for key, strategy_config in config.items():
            if not isinstance(strategy_config, dict):
                model = strategy_config
                kwargs = {"model": model}
                cls = QuickKerasFilter
            else:
                if (
                    "type" not in strategy_config
                    or strategy_config["type"] == "quick-filter"
                ):
                    cls = QuickKerasFilter
                else:
                    strategy_spec = FILTER_STRATEGY_ALIAS.get(strategy_config["type"], strategy_config["type"])
                    cls = load_dynamic_class(strategy_spec, filter_strategy_module, PolicyException)
                kwargs = dict(strategy_config)

            if "type" in kwargs:
                del kwargs["type"]
            obj = cls(key, self._config, **kwargs)
            self.load(obj)

    def reset_cache(self):
        """Reset filtering cache."""
        if not self.selection:
            return

        for name in self.selection:
            if hasattr(self[name], "reset_cache"):
                self[name].reset_cache()
