from pydantic import BaseModel

from shallowtree.configs.cache_configuration import CacheConfiguration
from shallowtree.configs.expansion_configuration import ExpansionConfiguration
from shallowtree.configs.filter_configuration import FilterConfiguration
from shallowtree.configs.search_configuration import SearchConfiguration
from shallowtree.configs.stock_configuration import StockConfiguration


class ApplicationConfiguration(BaseModel):
    search: SearchConfiguration
    expansion: ExpansionConfiguration
    filter: FilterConfiguration
    stock: StockConfiguration
    cache: CacheConfiguration