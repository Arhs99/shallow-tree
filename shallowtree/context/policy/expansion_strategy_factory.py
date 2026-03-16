from typing import Any
from shallowtree.context.expansion_strategies.template_based_expansion_strategy import TemplateBasedExpansionStrategy
from shallowtree.utils.exceptions import PolicyException
from shallowtree.context.expansion_strategies.expansion_strategies import (
    __name__ as expansion_strategy_module, ExpansionStrategy,
)
from shallowtree.utils.loading import load_dynamic_class


class ExpansionStrategyFactory:

    @staticmethod
    def load_from_config(**config: Any) -> ExpansionStrategy:
        """
        Load one or more expansion policy from a configuration

        The format should be
        key:
            type: name of the expansion class or custom_package.custom_model.CustomClass
            model: path_to_model
            template: path_to_templates
            other settings or params
        or
        key:
            - path_to_model
            - path_to_templates

        :param config: the configuration
        """
        for key, strategy_config in config.items():
            if not isinstance(strategy_config, dict):
                model, template = strategy_config
                kwargs = {"model": model, "template": template}
                cls = TemplateBasedExpansionStrategy
            else:
                if "type" not in strategy_config or strategy_config["type"] == "template-based":
                    cls = TemplateBasedExpansionStrategy
                else:
                    cls = load_dynamic_class(
                        strategy_config["type"],
                        expansion_strategy_module,
                        PolicyException,
                    )
                kwargs = dict(strategy_config)

            if "type" in kwargs:
                del kwargs["type"]
            obj = cls(key, None, **kwargs) #self._config
            return obj