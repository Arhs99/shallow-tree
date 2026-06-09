from shallowtree.configs.input_configuration import InputConfiguration
from shallowtree.interfaces.search_modes.base_tree_search import BaseTreeSearch
from shallowtree.interfaces.search_modes.scaffold_search import ScaffoldSearch
from shallowtree.interfaces.search_modes.standard_search import StandardSearch


class TreeSearch:
    def __new__(cls, input_config: InputConfiguration) -> BaseTreeSearch:
        if input_config.scaffold:
            return ScaffoldSearch(input_config)
        else:
            return StandardSearch(input_config)