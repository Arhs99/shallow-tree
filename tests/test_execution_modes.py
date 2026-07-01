import json
import tempfile
import unittest
from pathlib import Path

from shallowtree.configs.application_configuration import ApplicationConfiguration
from shallowtree.configs.input_configuration import InputConfiguration
from shallowtree.context.config import Configuration
from shallowtree.interfaces.execution_modes import parallel_search, sequential_search, iterative_deepening_search, \
    parallel_iterative_deepening_search

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "application_config/config.json"


class TestExecutionModes(unittest.TestCase):

    def setUp(self):

        config = json.loads(CONFIG_PATH.read_text())
        config.setdefault("cache", {})["enabled"] = False
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(config, tmp)
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        config_path = tmp.name

        smiles = [
            "Clc1ccccc1COC5CC(Nc3n[nH]c4cc(c2ccccc2)ccc34)C5",
            "CC(c2c[nH]c3cc(c1ccccc1)ccc23)C5CC(OCc4ccccc4Cl)C5",
            "CNC(=O)c1nn(C)c2c1C(C)(C)Cc1cnc(Nc3ccc(CN4CCN(C)CC4)cc3)nc1-2",
            "CC(C)(C)c1cc2c(N/N=C\\c3cccc(CN)n3)ncnc2s1",
            "COc1cccc2c1c(Cl)c1c3c(cc(O)c(O)c32)C(=O)N1",
            # "CCCOc1cccc2c1c(Cl)c1c3c(cc(O)c(O)c32)C(=O)N1",
            # "CCOc1cccc2c1c(Cl)c1c3c(cc(O)c(O)c32)C(=O)N1",
        ]
        self.standard_config = InputConfiguration(app_configuration_path=config_path,
                                         scaffold=None,
                                         routes=True, depth=2, smiles=smiles, output_path="", parallel_processes=3,
                                                  d_max=2)
        self.scaffold_config = InputConfiguration(app_configuration_path=config_path,
                                         scaffold="[*]c1n[nH]c2cc(-c3ccccc3)ccc12",
                                         routes=True, depth=2, smiles=smiles, output_path="", parallel_processes=3,
                                                  d_max=2)
        self.iterative_config = InputConfiguration(app_configuration_path=config_path,
                                                  scaffold=None,
                                                  routes=True, depth=2, smiles=smiles, output_path="",
                                                  parallel_processes=2, d_max=2)

    def test_parallel_standard_search(self):
        # expected = [['Clc1n[nH]c2cc(-c3ccccc3)ccc12', 'Clc1ccccc1CBr', 'NC1CC(O)C1'], [], [], ['O=Cc1cccc(CO)n1', 'CC(C)(C)Cl', 'NNc1ncnc2sccc12'], []]
        expected = [['Clc1ccccc1CBr', 'O=C1CC(O)C1', 'Nc1n[nH]c2cc(Br)ccc12', 'OB(O)c1ccccc1'], [], [],
                    ['O=Cc1cccc(CO)n1', 'CC(C)(C)Cl', 'NNc1ncnc2sccc12'], []]
        expected_scores = [1.0, 0.9375, 0.875, 1.0, 0.5]
        df_result = parallel_search(self.standard_config)
        result = df_result["BBs"].tolist()
        scores = df_result["score"].tolist()
        print(df_result)
        self.assertListEqual(expected, result)
        self.assertListEqual(expected_scores, scores)

    def test_parallel_scaffold_search(self):
        # expected = [['Clc1n[nH]c2cc(-c3ccccc3)ccc12', 'Clc1ccccc1CBr', 'NC1CC(O)C1'], [], [], ['O=Cc1cccc(CO)n1', 'CC(C)(C)Cl', 'NNc1ncnc2sccc12'], []]
        expected = [['Clc1n[nH]c2cc(-c3ccccc3)ccc12', 'Clc1ccccc1CBr', 'NC1CC(O)C1'],[],[],[],[]]
        expected_scores = [1,0,0,0,0]
        df_result = parallel_search(self.scaffold_config)
        result = df_result["BBs"].tolist()
        scores = df_result["score"].tolist()
        print(df_result)
        self.assertListEqual(expected, result)
        self.assertListEqual(expected_scores, scores)


    def test_sequential_scaffold_search(self):
        expected = [['Clc1n[nH]c2cc(-c3ccccc3)ccc12', 'Clc1ccccc1CBr', 'NC1CC(O)C1'],[],[],[],[]]
        expected_scores = [1,0,0,0,0]
        df_result = sequential_search(self.scaffold_config)
        result = df_result["BBs"].tolist()
        scores = df_result["score"].tolist()
        print(df_result)
        self.assertListEqual(expected, result)
        self.assertListEqual(expected_scores, scores)


    def test_sequential_standard_search(self):
        expected = [['Clc1ccccc1CBr', 'O=C1CC(O)C1', 'Nc1n[nH]c2cc(Br)ccc12', 'OB(O)c1ccccc1'], [], [],
                    ['O=Cc1cccc(CO)n1', 'CC(C)(C)Cl', 'NNc1ncnc2sccc12'], []]
        expected_scores = [1.0, 0.9375, 0.875, 1.0, 0.5]
        df_result = sequential_search(self.standard_config)
        result = df_result["BBs"].tolist()
        scores = df_result["score"].tolist()
        print(df_result)
        self.assertListEqual(expected, result)
        self.assertListEqual(expected_scores, scores)

    def test_iterative_deepening_search(self):
        expected = [['Clc1ccccc1CBr', 'O=C1CC(O)C1', 'Nc1n[nH]c2cc(Br)ccc12', 'OB(O)c1ccccc1'], [], [],
                    ['O=Cc1cccc(CO)n1', 'CC(C)(C)Cl', 'NNc1ncnc2sccc12'], []]
        expected_scores = [1.0, 0.9375, 0.875, 1.0, 0.5]
        df_result = parallel_iterative_deepening_search(self.standard_config)
        result = df_result["BBs"].tolist()
        scores = df_result["score"].tolist()
        print(df_result)
        self.assertListEqual(expected, result)
        self.assertListEqual(expected_scores, scores)

    def test_iterative_scaffold_deepening_search(self):
        expected = [['Clc1n[nH]c2cc(-c3ccccc3)ccc12', 'Clc1ccccc1CBr', 'NC1CC(O)C1'], [], [], [], []]
        expected_scores = [1, 0, 0, 0, 0]
        df_result = parallel_iterative_deepening_search(self.scaffold_config)
        result = df_result["BBs"].tolist()
        scores = df_result["score"].tolist()
        print(df_result)
        self.assertListEqual(expected, result)
        self.assertListEqual(expected_scores, scores)