import copy
from typing import List

import pandas as pd
from pathos.multiprocessing import ProcessPool

from shallowtree.configs.input_configuration import InputConfiguration

from shallowtree.configs.application_configuration import ApplicationConfiguration
from shallowtree.context.config import Configuration
from shallowtree.interfaces.full_tree_search import Expander


def clone_config(input_config: InputConfiguration, smiles:List[str]):
    clone = copy.deepcopy(input_config)
    clone.smiles = smiles
    return clone

def standard_search(input_config: InputConfiguration):
    config_dict = Configuration.from_json(input_config.configuration_yml_path)
    app_config = ApplicationConfiguration(**config_dict)
    smiles = [input_config.smiles[i:i + input_config.parallel_processes]
              for i in range(0, len(input_config.smiles), input_config.parallel_processes)]
    input_configs = [clone_config(input_config, s) for s in smiles]

    def parallel_run(input_c):
        expander = Expander(app_config)
        expander.expansion_policy.select_first()
        expander.filter_policy.select_first()
        expander.stock.select_first()
        df = expander.search_tree(input_c.smiles, max_depth=input_c.depth)
        if not input_c.routes:
            df = df.drop(columns=['route'])
        return df

    pool = ProcessPool(nodes=input_config.parallel_processes)
    results = pool.map(parallel_run, input_configs)
    concatenated = pd.concat(results)
    return concatenated


def scaffold_search(input_config: InputConfiguration):
    config_dict = Configuration.from_json(input_config.configuration_yml_path)
    app_config = ApplicationConfiguration(**config_dict)
    smiles = [input_config.smiles[i:i + input_config.parallel_processes]
              for i in range(0, len(input_config.smiles), input_config.parallel_processes)]
    input_configs = [clone_config(input_config, s) for s in smiles]

    def parallel_run(input_c):
        expander = Expander(app_config)
        expander.expansion_policy.select_first()
        expander.filter_policy.select_first()
        expander.stock.select_first()
        df = expander.context_search(input_c.smiles, input_c.scaffold, max_depth=input_c.depth)
        if not input_c.routes:
            df = df.drop(columns=['route'])
        return df

    pool = ProcessPool(nodes=input_config.parallel_processes)
    results = pool.map(parallel_run, input_configs)
    concatenated = pd.concat(results)
    return concatenated


def sequential_search(input_config: InputConfiguration):
    config_dict = Configuration.from_json(input_config.configuration_yml_path)
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