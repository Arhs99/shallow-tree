""" Module containing a class for encapsulating the settings of the tree search
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import yaml

from shallowtree.context.policy import ExpansionPolicy, FilterPolicy
from shallowtree.context.stock import Stock
from shallowtree.utils.logging import logger

if TYPE_CHECKING:
    from shallowtree.utils.type_utils import Any, Dict, List, Optional, StrDict, Union

#
# @dataclass
# class _PostprocessingConfiguration:
#     min_routes: int = 5
#     max_routes: int = 25
#     all_routes: bool = False
#     route_distance_model: Optional[str] = None
#     route_scorers: List[str] = field(default_factory=lambda: [])
#     scorer_weights: Optional[List[float]] = field(default_factory=lambda: None)
#
#
# @dataclass
# class _SearchConfiguration:
#     algorithm: str = "mcts"
#     algorithm_config: Dict[str, Any] = field(
#         default_factory=lambda: {
#             "C": 1.4,
#             "default_prior": 0.5,
#             "use_prior": True,
#             "prune_cycles_in_search": True,
#             "search_rewards": ["state score"],
#             "immediate_instantiation": (),
#             "mcts_grouping": None,
#             "search_rewards_weights": [],
#         }
#     )
#     max_transforms: int = 6
#     iteration_limit: int = 100
#     time_limit: int = 120
#     return_first: bool = False
#     exclude_target_from_stock: bool = True
#     break_bonds: List[List[int]] = field(default_factory=list)
#     freeze_bonds: List[List[int]] = field(default_factory=list)
#     break_bonds_operator: str = "and"


@dataclass
class Configuration:
    """
    Encapsulating the settings of the tree search, including the policy,
    the stock, the loaded scorers and various parameters.
    """

    stock: Stock = field(init=False)
    expansion_policy: ExpansionPolicy = field(init=False)
    filter_policy: FilterPolicy = field(init=False)

    def __post_init__(self) -> None:
        self.stock = Stock()
        self.expansion_policy = ExpansionPolicy(self)
        self.filter_policy = FilterPolicy(self)
        self._logger = logger()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Configuration):
            return False
        for key, setting in vars(self).items():
            if isinstance(setting, (int, float, str, bool, list)):
                if (
                    vars(self)[key] != vars(other)[key]
                ):
                    return False
        return True

    @classmethod
    def from_dict(cls, source: StrDict) -> "Configuration":
        """
        Loads a configuration from a dictionary structure.
        The parameters not set in the dictionary are taken from the default values.
        The policies and stocks specified are directly loaded.

        :param source: the dictionary source
        :return: a Configuration object with settings from the source
        """
        expansion_config = source.pop("expansion", {})
        filter_config = source.pop("filter", {})
        stock_config = source.pop("stock", {})

        config_obj = Configuration()

        config_obj.expansion_policy.load_from_config(**expansion_config)
        config_obj.filter_policy.load_from_config(**filter_config)
        config_obj.stock.load_from_config(**stock_config)

        return config_obj

    @classmethod
    def from_file(cls, filename: str) -> "Configuration":
        """
        Loads a configuration from a yaml file.
        The parameters not set in the yaml file are taken from the default values.
        The policies and stocks specified in the yaml file are directly loaded.
        The parameters in the yaml file may also contain environment variables as
        values.

        :param filename: the path to a yaml file
        :return: a Configuration object with settings from the yaml file
        :raises:
            ValueError: if parameter's value expects an environment variable that
                does not exist in the current environment
        """
        with open(filename, "r") as fileobj:
            txt = fileobj.read()
        environ_var = re.findall(r"\$\{.+?\}", txt)
        for item in environ_var:
            if item[2:-1] not in os.environ:
                raise ValueError(f"'{item[2:-1]}' not in environment variables")
            txt = txt.replace(item, os.environ[item[2:-1]])
        _config = yaml.load(txt, Loader=yaml.SafeLoader)
        return Configuration.from_dict(_config)

    def _update_from_config(self, config: StrDict) -> None:
        return
