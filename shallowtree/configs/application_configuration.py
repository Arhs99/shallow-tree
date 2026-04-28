from typing import List

from pydantic import BaseModel

from shallowtree.configs.cache_configuration import CacheConfiguration
from shallowtree.configs.expansion_configuration import ExpansionConfiguration
from shallowtree.configs.filter_configuration import FilterConfiguration
from shallowtree.configs.search_configuration import SearchConfiguration
from shallowtree.configs.stock_configuration import StockConfiguration


class ApplicationConfiguration(BaseModel):
    search: SearchConfiguration
    expansion: List[ExpansionConfiguration]
    filter: List[FilterConfiguration]
    stock: List[StockConfiguration]
    cache: CacheConfiguration
    extra_template_path: str|None = None