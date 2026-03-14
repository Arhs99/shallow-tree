import unittest

from shallowtree.configs.input_configuration import InputConfiguration
from shallowtree.interfaces.full_tree_search import Expander


class TestMultiDimensions(unittest.TestCase):

    def setUp(self):
        smiles = [
            "Clc1ccccc1COC5CC(Nc3n[nH]c4cc(c2ccccc2)ccc34)C5",
            "CC(c2c[nH]c3cc(c1ccccc1)ccc23)C5CC(OCc4ccccc4Cl)C5"
	    ]
        self.config = InputConfiguration(configuration_yml_path="/home/patronov/data/synth/config.yml",
                                         scaffold="[*]c1n[nH]c2cc(-c3ccccc3)ccc12",
                                         routes=True, depth=2, smiles=smiles, output_path="")
        self.expander = Expander(configfile=self.config.configuration_yml_path)
        self.expander.expansion_policy.select_first()
        self.expander.filter_policy.select_first()
        self.expander.stock.select_first()

    def test_1(self):
        df = self.expander.context_search(self.config.smiles, self.config.scaffold, max_depth=self.config.depth)
        df = df.drop(columns=['route'])
        print(df.head())