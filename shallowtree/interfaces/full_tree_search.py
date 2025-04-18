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

from collections import defaultdict
from typing import TYPE_CHECKING

from rdkit import Chem
import pandas as pd

from shallowtree.chem import Molecule, TreeMolecule
from shallowtree.context.config import Configuration
from shallowtree.context.policy.expansion_strategies import TemplateRules

# This must be imported first to setup logging for rdkit, tensorflow etc
from shallowtree.utils.logging import logger

if TYPE_CHECKING:
    from shallowtree.utils.type_utils import (
        List,
        Optional,
        StrDict,
    )

extra_template_path = '/home/kostas/dev/shallow-tree/shallowtree/rules/direct.csv'

class Expander:
    """
    """

    def __init__(
            self, configfile: Optional[str] = None, configdict: Optional[StrDict] = None
    ):
        self._logger = logger()

        if configfile:
            self.config = Configuration.from_file(configfile)
        elif configdict:
            self.config = Configuration.from_dict(configdict)
        else:
            self.config = Configuration()

        self.expansion_policy = self.config.expansion_policy
        self.rules_expansion = TemplateRules(extra_template_path)
        self.filter_policy = self.config.filter_policy
        self.stock = self.config.stock
        self.max_depth = 2

    def context_search(self, smiles: List[str], scaffold_str: str, max_depth=2) -> pd.DataFrame:
        self.max_depth = max_depth
        rows = []
        for smi in smiles:
            solution = defaultdict(list)
            mol = TreeMolecule(parent=None, smiles=smi)
            scaffold = Chem.MolFromSmarts(scaffold_str)
            self.solved = dict()
            self.BBs = []
            self.cache = dict()
            self._counter = 0
            self._cache_counter = 0
            score = 0.0
            actions, _ = self.expansion_policy.get_actions([mol])
            det_actions = self.rules_expansion.get_actions([mol])
            for action in det_actions + actions:
                reactants = action.reactants
                feasibility_prob = 0
                if not reactants:
                    continue
                if action.metadata['policy_name'] == 'rules':
                    feasibility_prob = 1.0
                else:
                    for name in self.filter_policy.selection:
                        _, feasibility_prob = self.filter_policy[name].feasibility(action)
                        action.metadata["feasibility"] = float(feasibility_prob)
                        break
                if feasibility_prob < 0.5:
                    continue
                root_match = set(mol.index_to_mapping[x] for x in mol.rd_mol.GetSubstructMatch(scaffold))
                for r in reactants[0]:
                    r_match = set(r.index_to_mapping[x] for x in r.rd_mol.GetSubstructMatch(scaffold))
                    if r_match and len(r_match ^ root_match) == 2:
                        score = sum([self.req_search_tree(x, 1) for x in reactants[0] if x != r]) / (len(
                            reactants[0]) - 1)
                        if score > 0.9:
                            self.solved[mol.inchi_key] = (reactants[0], score, action.metadata['classification'])
                        break
            if score > 0.9:
                self.best_route(mol, 0, solution)
                # json_data = json.dumps(dict(solution), indent=2)
            rows.append({'SMILES': smi, 'score': score, 'route': dict(solution), 'BBs': self.BBs})
        df = pd.DataFrame(rows)
        return df

    def search_tree(
            self,
            smiles: List[str],
            max_depth=2
    ) -> pd.DataFrame:
        """
        """
        self.max_depth = max_depth
        rows = []
        for smi in smiles:
            solution = defaultdict(list)
            mol = TreeMolecule(parent=None, smiles=smi)
            self.solved = dict()
            self.BBs = []
            self.cache = dict()
            self._counter = 0
            self._cache_counter = 0
            score = self.req_search_tree(mol, depth=0)
            if score > 0.9:
                self.best_route(mol, 0, solution)
                # json_data = json.dumps(dict(solution), indent=2)
            rows.append({'SMILES': smi, 'score': score, 'route': dict(solution), 'BBs': self.BBs})
        df = pd.DataFrame(rows)
        return df

    def req_search_tree(self, mol: TreeMolecule, depth: int) -> float:
        if depth > self.max_depth:
            return 0.0
        self._counter += 1
        if mol in self.stock:
            return 1.0
        if mol.inchi_key in self.cache.keys():
            self._cache_counter += 1
            cdepth, cscore = self.cache[mol.inchi_key]
            if cdepth <= depth:
                return cscore

        actions, _ = self.expansion_policy.get_actions([mol])
        det_actions = self.rules_expansion.get_actions([mol])
        score = 0.0
        for action in actions + det_actions:
            reactants = action.reactants
            feasibility_prob = 0
            if not reactants:
                continue
            if action.metadata['policy_name'] == 'rules':
                feasibility_prob = 1.0
            else:
                for name in self.filter_policy.selection:
                    _, feasibility_prob = self.filter_policy[name].feasibility(action)
                    action.metadata["feasibility"] = float(feasibility_prob)
                    break
            if feasibility_prob < 0.5:
                continue
            score = sum([self.req_search_tree(x, depth + 1) for x in reactants[0]]) / len(reactants[0])
            if score > 0.9:
                self.solved[mol.inchi_key] = (reactants[0], score, action.metadata['classification'])
                self.cache[mol.inchi_key] = (depth, score)
                return score
        self.cache[mol.inchi_key] = (depth, score)
        return score

    def best_route(self, mol: Molecule, depth: int, tree: defaultdict):
        while depth <= self.max_depth:
            tup = self.solved.get(mol.inchi_key)
            if tup is None:
                self.BBs.append(mol.smiles)
                return
            else:
                rxn, score, clas = tup
                reactants = '.'.join([m.smiles for m in rxn])
                tree[depth + 1].append([f'{mol.smiles} => {reactants}', clas])
                for x in rxn:
                    self.best_route(x, depth + 1, tree)
                return
