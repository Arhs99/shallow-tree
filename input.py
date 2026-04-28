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

from rdkit import Chem

from shallowtree.configs.input_configuration import InputConfiguration
from shallowtree.interfaces.parallel import sequential_search, scaffold_search, standard_search


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

    if input_config.parallel_processes ==1:
        df = sequential_search(input_config)
    else:
        if input_config.scaffold is not None:
            df = scaffold_search(input_config)
        else:
            df = standard_search(input_config)

    if not input_config.routes:
        df = df.drop(columns=['route'])

    df.to_csv(input_config.output_path, index=False)  # NOT for very large files that can overflow the stdout buffer, can use chunks etc


if __name__ == '__main__':
    main()
