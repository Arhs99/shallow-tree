import unittest
from os import path

from shallowtree.configs.application_configuration import ApplicationConfiguration
from shallowtree.configs.input_configuration import InputConfiguration
from shallowtree.context.config import Configuration
from shallowtree.interfaces.full_tree_search import Expander
from pathos.pools import ProcessPool


class TestParallelRuns(unittest.TestCase):

    def setUp(self):
        smiles = [
            "Clc1ccccc1COC5CC(Nc3n[nH]c4cc(c2ccccc2)ccc34)C5",
            "CC(c2c[nH]c3cc(c1ccccc1)ccc23)C5CC(OCc4ccccc4Cl)C5",
            "CNC(=O)c1nn(C)c2c1C(C)(C)Cc1cnc(Nc3ccc(CN4CCN(C)CC4)cc3)nc1-2",
            "CC(C)(C)c1cc2c(N/N=C\c3cccc(CN)n3)ncnc2s1",
            "COc1cccc2c1c(Cl)c1c3c(cc(O)c(O)c32)C(=O)N1",
	    ]
        self.config = InputConfiguration(configuration_yml_path="/home/patronov/data/synth/config.json",
                                         scaffold="[*]c1n[nH]c2cc(-c3ccccc3)ccc12",
                                         routes=True, depth=2, smiles=smiles, output_path="")
        # self.expander = Expander(configfile=self.config.configuration_yml_path)
        # self.expander.expansion_policy.select_first()
        # self.expander.filter_policy.select_first()
        # self.expander.stock.select_first()

    def test_context_search(self):
        df = self.expander.context_search(self.config.smiles, self.config.scaffold, max_depth=self.config.depth)
        df = df.drop(columns=['route'])
        print(df.head())

    def test_search_tree(self):
        df = self.expander.search_tree(self.config.smiles, max_depth=self.config.depth)
        df = df.drop(columns=['route'])
        print(df.head())

    def test_parallel(self):
        smiles = [
            "Clc1ccccc1COC5CC(Nc3n[nH]c4cc(c2ccccc2)ccc34)C5",
            "CC(c2c[nH]c3cc(c1ccccc1)ccc23)C5CC(OCc4ccccc4Cl)C5",
            "CNC(=O)c1nn(C)c2c1C(C)(C)Cc1cnc(Nc3ccc(CN4CCN(C)CC4)cc3)nc1-2",
            "CC(C)(C)c1cc2c(N/N=C\c3cccc(CN)n3)ncnc2s1",
            "COc1cccc2c1c(Cl)c1c3c(cc(O)c(O)c32)C(=O)N1",
        ]
        config = InputConfiguration(configuration_yml_path="/home/patronov/data/synth/config.json",
                                    scaffold="[*]c1n[nH]c2cc(-c3ccccc3)ccc12",
                                    routes=True, depth=2, smiles=smiles, output_path="")
        config_dict = Configuration.from_json(config.configuration_yml_path)
        app_config = ApplicationConfiguration(**config_dict)

        def parallel_run(configuration):
            expander = Expander(configuration)
            expander.expansion_policy.select_first()
            expander.filter_policy.select_first()
            expander.stock.select_first()
            df = expander.context_search(config.smiles, config.scaffold, max_depth=config.depth)
            df = df.drop(columns=['route'])
            print(df.head())

        pool = ProcessPool(nodes=2)
        mapped_pool = pool.map(parallel_run, [app_config, app_config])
        mapped_pool.clear()