import unittest
from pathlib import Path

from shallowtree.configs.application_configuration import ApplicationConfiguration
from shallowtree.configs.input_configuration import InputConfiguration
from shallowtree.context.config import Configuration
from shallowtree.interfaces.parallel import standard_search, scaffold_search, sequential_search

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestParallelRuns(unittest.TestCase):

    def setUp(self):
        smiles = [
            "Clc1ccccc1COC5CC(Nc3n[nH]c4cc(c2ccccc2)ccc34)C5",
            "CC(c2c[nH]c3cc(c1ccccc1)ccc23)C5CC(OCc4ccccc4Cl)C5",
            "CNC(=O)c1nn(C)c2c1C(C)(C)Cc1cnc(Nc3ccc(CN4CCN(C)CC4)cc3)nc1-2",
            "CC(C)(C)c1cc2c(N/N=C\\c3cccc(CN)n3)ncnc2s1",
            "COc1cccc2c1c(Cl)c1c3c(cc(O)c(O)c32)C(=O)N1",
        ]
        self.config = InputConfiguration(app_configuration_path=str(REPO_ROOT / "application_config/config.json"),
                                         scaffold="[*]c1n[nH]c2cc(-c3ccccc3)ccc12",
                                         # scaffold=None,
                                         routes=True, depth=2, smiles=smiles, output_path="", parallel_processes=3)
        config_dict = Configuration.from_json(self.config.app_configuration_path)
        self.app_config = ApplicationConfiguration(**config_dict)

    def test_parallel_scaffold_search(self):
        expected = [['Clc1n[nH]c2cc(-c3ccccc3)ccc12', 'Clc1ccccc1CBr', 'NC1CC(O)C1'], [], [], [], []]
        expected_scores = [1, 0, 0, 0, 0]
        config = self.config
        df_result = scaffold_search(config)
        result = df_result["BBs"].tolist()
        scores = df_result["score"].tolist()
        self.assertListEqual(expected, result)
        self.assertListEqual(expected_scores, scores)
        print(result)

    def test_parallel_standard_search(self):
        expected = [['Clc1n[nH]c2cc(-c3ccccc3)ccc12', 'Clc1ccccc1CBr', 'NC1CC(O)C1'], [], [], ['O=Cc1cccc(CO)n1', 'CC(C)(C)Cl', 'NNc1ncnc2sccc12'], []]
        expected_scores = [1, 0.5, 0.75, 1.0, 0]
        config = self.config
        df_result = standard_search(config)
        result = df_result["BBs"].tolist()
        scores = df_result["score"].tolist()
        self.assertListEqual(expected, result)
        self.assertListEqual(expected_scores, scores)
        print(df_result)

    def test_sequential_search(self):
        expected = [['Clc1n[nH]c2cc(-c3ccccc3)ccc12', 'Clc1ccccc1CBr', 'NC1CC(O)C1'],[],[],[],[]]
        expected_scores = [1,0,0,0,0]
        config = self.config
        df_result = sequential_search(config)
        result = df_result["BBs"].tolist()
        scores = df_result["score"].tolist()
        self.assertListEqual(expected, result)
        self.assertListEqual(expected_scores, scores)
        print(df_result)