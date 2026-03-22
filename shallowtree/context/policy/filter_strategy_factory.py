from typing import Any

from shallowtree.context.filters.quick_keras_filter import QuickKerasFilter
from shallowtree.utils.exceptions import PolicyException
from shallowtree.utils.loading import load_dynamic_class
from shallowtree.context.filters.filter_strategy import (
    __name__ as filter_strategy_module, FILTER_STRATEGY_ALIAS, FilterStrategy,
)


class FilterStrategyFactory:

    @staticmethod
    def load_from_config(**config: Any) -> FilterStrategy:
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
                if ("type" not in strategy_config or strategy_config["type"] == "quick-filter"):
                    cls = QuickKerasFilter
                else:
                    strategy_spec = FILTER_STRATEGY_ALIAS.get(strategy_config["type"], strategy_config["type"])
                    cls = load_dynamic_class(strategy_spec, filter_strategy_module, PolicyException)
                kwargs = dict(strategy_config)

            if "type" in kwargs:
                del kwargs["type"]
            obj = cls(key, None, **kwargs) #self._config
            return obj