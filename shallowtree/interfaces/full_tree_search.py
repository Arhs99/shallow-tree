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
from datetime import datetime
from pathlib import Path
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

# TODO: Move extra_template_path to config.yml instead of hard-coding
extra_template_path = Path(__file__).parent.parent / 'rules' / 'direct.csv'

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
        self.redis_cache = self.config.redis_cache
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

            # Pre-populate from Redis if available
            if self.redis_cache:
                self._load_from_redis(mol)

            score = 0.0
            actions, _ = self.expansion_policy.get_actions([mol])
            det_actions = self.rules_expansion.get_actions([mol])

            # Separate rules-based from ML-based actions, filtering out empty reactants
            rules_actions = []
            ml_actions = []
            for action in det_actions + actions:
                if not action.reactants:
                    continue
                if action.metadata['policy_name'] == 'rules':
                    action.metadata["feasibility"] = 1.0
                    rules_actions.append(action)
                else:
                    ml_actions.append(action)

            # Batch predict all ML actions at once
            ml_feasibilities = []
            if ml_actions:
                filter_name = next(iter(self.filter_policy.selection))
                ml_feasibilities = self.filter_policy[filter_name].batch_feasibility(ml_actions)
                for action, (_, prob) in zip(ml_actions, ml_feasibilities):
                    action.metadata["feasibility"] = prob

            # Collect all feasible actions (rules + ML with prob >= 0.5)
            feasible_actions = rules_actions + [
                action for action, (_, prob) in zip(ml_actions, ml_feasibilities)
                if prob >= 0.5
            ]

            root_match = set(mol.index_to_mapping[x] for x in mol.rd_mol.GetSubstructMatch(scaffold))
            for action in feasible_actions:
                reactants = action.reactants
                for r in reactants[0]:
                    r_match = set(r.index_to_mapping[x] for x in r.rd_mol.GetSubstructMatch(scaffold))
                    if r_match and len(r_match ^ root_match) == 2:
                        score = sum([self.req_search_tree(x, 1) for x in reactants[0] if x != r]) / (len(
                            reactants[0]) - 1)
                        if score > 0.9:
                            self.solved[mol.inchi_key] = (reactants[0], score, action.metadata['classification'])
                        break
            # Persist to Redis if available and successful
            if self.redis_cache and score > 0.9:
                self._save_to_redis()

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

            # Pre-populate from Redis if available
            if self.redis_cache:
                self._load_from_redis(mol)

            score = self.req_search_tree(mol, depth=0)

            # Persist to Redis if available and successful
            if self.redis_cache and score > 0.9:
                self._save_to_redis()

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

        # Check Redis cache if local cache miss
        if self.redis_cache and mol.inchi_key not in self.cache:
            redis_cache_data = self.redis_cache.get_cache(mol.inchi_key)
            if redis_cache_data:
                cdepth, cscore = redis_cache_data
                self.cache[mol.inchi_key] = (cdepth, cscore)  # Populate local cache
                if cdepth <= depth:
                    self._cache_counter += 1
                    # Also try to load solved data
                    solved_data = self.redis_cache.get_solved(mol.inchi_key)
                    if solved_data:
                        self.solved[mol.inchi_key] = solved_data
                    return cscore

        actions, _ = self.expansion_policy.get_actions([mol])
        det_actions = self.rules_expansion.get_actions([mol])

        # Separate rules-based from ML-based actions, filtering out empty reactants
        rules_actions = []
        ml_actions = []
        for action in actions + det_actions:
            if not action.reactants:
                continue
            if action.metadata['policy_name'] == 'rules':
                action.metadata["feasibility"] = 1.0
                rules_actions.append(action)
            else:
                ml_actions.append(action)

        # Batch predict all ML actions at once
        ml_feasibilities = []
        if ml_actions:
            filter_name = next(iter(self.filter_policy.selection))
            ml_feasibilities = self.filter_policy[filter_name].batch_feasibility(ml_actions)
            for action, (_, prob) in zip(ml_actions, ml_feasibilities):
                action.metadata["feasibility"] = prob

        # Collect all feasible actions (rules + ML with prob >= 0.5)
        feasible_actions = rules_actions + [
            action for action, (_, prob) in zip(ml_actions, ml_feasibilities)
            if prob >= 0.5
        ]

        score = 0.0
        for action in feasible_actions:
            reactants = action.reactants
            score = sum([self.req_search_tree(x, depth + 1) for x in reactants[0]]) / len(reactants[0])
            if score > 0.9:
                self.solved[mol.inchi_key] = (reactants[0], score, action.metadata['classification'])
                self.cache[mol.inchi_key] = (depth, score)
                if self.redis_cache:
                    self.redis_cache.set_cache(mol.inchi_key, depth, score)
                return score
        self.cache[mol.inchi_key] = (depth, score)
        if self.redis_cache:
            self.redis_cache.set_cache(mol.inchi_key, depth, score)
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

    def _load_from_redis(self, root_mol: TreeMolecule) -> None:
        """Pre-populate local caches from Redis for the root molecule subtree."""
        if not self.redis_cache:
            return

        cache_data = self.redis_cache.get_cache(root_mol.inchi_key)
        if cache_data:
            self.cache[root_mol.inchi_key] = cache_data
            solved_data = self.redis_cache.get_solved(root_mol.inchi_key)
            if solved_data:
                self.solved[root_mol.inchi_key] = solved_data
                self._load_solved_subtree(solved_data[0])

    def _load_solved_subtree(self, reactants: List[TreeMolecule]) -> None:
        """Recursively load solved entries for reactants."""
        for mol in reactants:
            if mol.inchi_key not in self.solved:
                solved_data = self.redis_cache.get_solved(mol.inchi_key)
                if solved_data:
                    self.solved[mol.inchi_key] = solved_data
                    self._load_solved_subtree(solved_data[0])

    def _save_to_redis(self) -> None:
        """Persist all solved routes to Redis."""
        if not self.redis_cache:
            return

        for inchi_key, (reactants, score, classification) in self.solved.items():
            self.redis_cache.set_solved(inchi_key, reactants, score, classification)
            if inchi_key in self.cache:
                depth, cache_score = self.cache[inchi_key]
                self.redis_cache.set_cache(inchi_key, depth, cache_score)
