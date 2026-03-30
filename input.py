# shalow-tree - Retrosynthetic analysis and scoring
# Copyright (C) 2025  Kostas Papadopoulos <kostasp97@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import json
import sys
from typing import List

from shallowtree.configs.application_configuration import ApplicationConfiguration
from shallowtree.configs.input_configuration import InputConfiguration
from shallowtree.context.config import Configuration
from shallowtree.interfaces.full_tree_search import Expander
import argparse

from rdkit import Chem


def read_json_file(path: str):
    with open(path) as f:
        json_input = f.read().replace('\r', '').replace('\n', '')
    try:
        return json.loads(json_input)
    except (ValueError, KeyError, TypeError) as e:
        print(f"JSON format error in file ${path}: \n ${e}")

def canonicalize_input(smiles_list: List[str]) -> List[str]:
    smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in smiles_list if Chem.MolFromSmiles(x) is not None]
    return smiles

def main():
    config_path = sys.argv[1]
    config_dict = read_json_file(config_path)
    input_config = InputConfiguration(**config_dict)
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

    df.to_csv(input_config.output_path, index=False)  # NOT for very large files that can overflow the stdout buffer, can use chunks etc


if __name__ == '__main__':
    main()
