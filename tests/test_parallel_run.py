import unittest
from os import path

from shallowtree.configs.application_configuration import ApplicationConfiguration
from shallowtree.configs.input_configuration import InputConfiguration
from shallowtree.context.config import Configuration
from shallowtree.interfaces.full_tree_search import Expander
from pathos.pools import ProcessPool

from shallowtree.interfaces.parallel import standard_search, scaffold_search


class TestParallelRuns(unittest.TestCase):

    def setUp(self):
        smiles = [
            "Clc1ccccc1COC5CC(Nc3n[nH]c4cc(c2ccccc2)ccc34)C5",
            "CC(c2c[nH]c3cc(c1ccccc1)ccc23)C5CC(OCc4ccccc4Cl)C5",
            "CNC(=O)c1nn(C)c2c1C(C)(C)Cc1cnc(Nc3ccc(CN4CCN(C)CC4)cc3)nc1-2",
            "CC(C)(C)c1cc2c(N/N=C\c3cccc(CN)n3)ncnc2s1",
            "COc1cccc2c1c(Cl)c1c3c(cc(O)c(O)c32)C(=O)N1",
        ]
        self.config = InputConfiguration(app_configuration_path="/home/patronov/data/synth/config.json",
                                         scaffold="[*]c1n[nH]c2cc(-c3ccccc3)ccc12",
                                         routes=True, depth=2, smiles=smiles, output_path="", parallel_processes=3)
        config_dict = Configuration.from_json(self.config.app_configuration_path)
        self.app_config = ApplicationConfiguration(**config_dict)

    def test_parallel_scaffold_search(self):
        config = self.config
        result = scaffold_search(config)
        print(result)

    def test_parallel_standard_search(self):
        config = self.config
        result = standard_search(config)
        print(result)

    def test_sequential_search(self):
        config = self.config
        print(vars(config))
        result = standard_search(config)
        print(result)