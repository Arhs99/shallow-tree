import copy
from typing import List

import pandas as pd
from pathos.multiprocessing import ProcessPool

from shallowtree.configs.input_configuration import InputConfiguration

from shallowtree.configs.application_configuration import ApplicationConfiguration
from shallowtree.context.config import Configuration
from shallowtree.context.stock.queries import InMemoryInchiKeyQuery
from shallowtree.context.stock.shared_inchi_key_set import SharedInchiKeySet
from shallowtree.context.stock.shared_stock_query import SharedInchiKeyQuery
from shallowtree.context.stock.stock import Stock
from shallowtree.interfaces.full_tree_search import Expander


def clone_config(input_config: InputConfiguration, smiles:List[str]):
    clone = copy.deepcopy(input_config)
    clone.smiles = smiles
    return clone


def _build_shared_stock(stock_configs) -> SharedInchiKeySet:
    """Load all stock inchi keys in the parent and pack them into a single
    SharedInchiKeySet. Workers attach to this block by name, paying for one
    physical copy in /dev/shm rather than one heap copy per worker."""
    keys: List[str] = []
    for cfg in stock_configs:
        q = InMemoryInchiKeyQuery(
            path=cfg.dataset,
            inchi_key_col=cfg.inchi_key_col,
            price_col=cfg.price_col,
        )
        keys.extend(q.stock_inchikeys)
        del q
    shared = SharedInchiKeySet.build(keys)
    del keys
    return shared


def _build_worker_stock(shm_name: str, shm_count: int) -> Stock:
    """Worker-side: attach to the shared inchi-key block and wrap it in a Stock."""
    attached = SharedInchiKeySet.attach(shm_name, shm_count)
    stock = Stock()
    stock.load(SharedInchiKeyQuery(attached), key="shared")
    return stock


def standard_search(input_config: InputConfiguration):
    config_dict = Configuration.from_json(input_config.app_configuration_path)
    app_config = ApplicationConfiguration(**config_dict)
    smiles = [input_config.smiles[i:i + input_config.parallel_processes]
              for i in range(0, len(input_config.smiles), input_config.parallel_processes)]
    input_configs = [clone_config(input_config, s) for s in smiles]

    shared = _build_shared_stock(app_config.stock)
    shm_name, shm_count = shared.shm_name, len(shared)

    def parallel_run(input_c):
        stock = _build_worker_stock(shm_name, shm_count)
        expander = Expander(app_config, prebuilt_stock=stock)
        expander.expansion_policy.select_first()
        expander.filter_policy.select_first()
        expander.stock.select_first()
        df = expander.search_tree(input_c.smiles, max_depth=input_c.depth)
        if not input_c.routes:
            df = df.drop(columns=['route'])
        return df

    try:
        pool = ProcessPool(nodes=input_config.parallel_processes)
        results = pool.map(parallel_run, input_configs)
        concatenated = pd.concat(results)
        return concatenated
    finally:
        shared.unlink()


def scaffold_search(input_config: InputConfiguration):
    config_dict = Configuration.from_json(input_config.app_configuration_path)
    app_config = ApplicationConfiguration(**config_dict)
    smiles = [input_config.smiles[i:i + input_config.parallel_processes]
              for i in range(0, len(input_config.smiles), input_config.parallel_processes)]
    input_configs = [clone_config(input_config, s) for s in smiles]

    shared = _build_shared_stock(app_config.stock)
    shm_name, shm_count = shared.shm_name, len(shared)

    def parallel_run(input_c):
        stock = _build_worker_stock(shm_name, shm_count)
        expander = Expander(app_config, prebuilt_stock=stock)
        expander.expansion_policy.select_first()
        expander.filter_policy.select_first()
        expander.stock.select_first()
        df = expander.context_search(input_c.smiles, input_c.scaffold, max_depth=input_c.depth)
        if not input_c.routes:
            df = df.drop(columns=['route'])
        return df

    try:
        pool = ProcessPool(nodes=input_config.parallel_processes)
        results = pool.map(parallel_run, input_configs)
        concatenated = pd.concat(results)
        return concatenated
    finally:
        shared.unlink()


def sequential_search(input_config: InputConfiguration):
    config_dict = Configuration.from_json(input_config.app_configuration_path)
    app_config = ApplicationConfiguration(**config_dict)

    expander = Expander(app_config)
    expander.expansion_policy.select_first()
    expander.filter_policy.select_first()
    expander.stock.select_first()

    if input_config.scaffold is None:
        df = expander.search_tree(input_config.smiles, max_depth=input_config.depth)
    else:
        df = expander.context_search(input_config.smiles, input_config.scaffold, max_depth=input_config.depth)

    if not input_config.routes:
        df = df.drop(columns=['route'])
    return df
