import unittest
from pathlib import Path

from shallowtree.configs.application_configuration import ApplicationConfiguration
from shallowtree.context.config import Configuration
from shallowtree.interfaces.full_tree_search import Expander

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestStandardRuns(unittest.TestCase):

    CONFIG_PATH = str(REPO_ROOT / "config.json")
    SCAFFOLD = "[*]c1n[nH]c2cc(-c3ccccc3)ccc12"
    DEPTH = 2

    def setUp(self):
        self.smiles = [
            "Clc1ccccc1COC5CC(Nc3n[nH]c4cc(c2ccccc2)ccc34)C5",
            "CC(c2c[nH]c3cc(c1ccccc1)ccc23)C5CC(OCc4ccccc4Cl)C5",
            "CNC(=O)c1nn(C)c2c1C(C)(C)Cc1cnc(Nc3ccc(CN4CCN(C)CC4)cc3)nc1-2",
            "CC(C)(C)c1cc2c(N/N=C\\c3cccc(CN)n3)ncnc2s1",
            "COc1cccc2c1c(Cl)c1c3c(cc(O)c(O)c32)C(=O)N1",
        ]
        config_dict = Configuration.from_json(self.CONFIG_PATH)
        app_config = ApplicationConfiguration(**config_dict)
        self.expander = Expander(app_config)
        self.expander.expansion_policy.select_first()
        self.expander.filter_policy.select_first()
        self.expander.stock.select_first()

    def test_context_search(self):
        df = self.expander.context_search(self.smiles, self.SCAFFOLD, max_depth=self.DEPTH)
        df = df.drop(columns=['route'])
        print(df.head())

    def test_search_tree(self):
        df = self.expander.search_tree(self.smiles, max_depth=self.DEPTH)
        df = df.drop(columns=['route'])
        print(df.head())