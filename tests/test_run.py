import unittest

from shallowtree.configs.input_configuration import InputConfiguration
from shallowtree.interfaces.full_tree_search import Expander


class TestStandardRuns(unittest.TestCase):

    def setUp(self):
        smiles = [
            "Clc1ccccc1COC5CC(Nc3n[nH]c4cc(c2ccccc2)ccc34)C5",
            "CC(c2c[nH]c3cc(c1ccccc1)ccc23)C5CC(OCc4ccccc4Cl)C5",
            "CNC(=O)c1nn(C)c2c1C(C)(C)Cc1cnc(Nc3ccc(CN4CCN(C)CC4)cc3)nc1-2",
            "CC(C)(C)c1cc2c(N/N=C\c3cccc(CN)n3)ncnc2s1",
            "COc1cccc2c1c(Cl)c1c3c(cc(O)c(O)c32)C(=O)N1",
	    ]
        self.config = InputConfiguration(app_configuration_path="/home/patronov/data/synth/config.yml",
                                         scaffold="[*]c1n[nH]c2cc(-c3ccccc3)ccc12",
                                         routes=True, depth=2, smiles=smiles, output_path="")
        self.expander = Expander(configfile=self.config.app_configuration_path)
        self.expander.expansion_policy.select_first()
        self.expander.filter_policy.select_first()
        self.expander.stock.select_first()

    def test_context_search(self):
        df = self.expander.context_search(self.config.smiles, self.config.scaffold, max_depth=self.config.depth)
        df = df.drop(columns=['route'])
        print(df.head())

    def test_search_tree(self):
        df = self.expander.search_tree(self.config.smiles, max_depth=self.config.depth)
        df = df.drop(columns=['route'])
        print(df.head())