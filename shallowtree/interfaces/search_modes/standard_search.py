
from __future__ import annotations

from collections import defaultdict
from typing import List

import pandas as pd
from rdkit.Chem.rdchem import Mol

from shallowtree.chem.molecules.tree_molecule import TreeMolecule
from shallowtree.interfaces.search_modes.base_tree_search import BaseTreeSearch


class StandardSearch(BaseTreeSearch):

    def search(self, smiles: List[str], max_depth=2) -> pd.DataFrame:
        self.max_depth = max_depth
        rows = []
        for smi in smiles:
            solution = defaultdict(list)
            mol = TreeMolecule(parent=None, smiles=smi)
            self.BBs = []

            self._load_from_redis(mol)
            score = self.req_search_tree(mol, depth=0)
            rows = self._update(mol, smi, score, solution, rows)

        df = pd.DataFrame(rows)
        return df

    def req_search_tree(self, mol: TreeMolecule, depth: int) -> float:
        if depth > self.max_depth:
            return 0.0

        if mol.inchi_key in self.cache:
            cdepth, cscore = self.cache[mol.inchi_key]
            if cdepth <= depth:
                return cscore

        if mol in self.stock:
            self.cache[mol.inchi_key] = (0, 1.0)
            return 1.0

        # Check Redis cache if local cache miss
        if self.redis_cache and mol.inchi_key not in self.cache:
            redis_cache_data = self.redis_cache.get_cache(mol.inchi_key)
            if redis_cache_data:
                cdepth, cscore = redis_cache_data
                self.cache[mol.inchi_key] = (cdepth, cscore)  # Populate local cache
                if cdepth <= depth:
                    # Also try to load solved data
                    solved_data = self.redis_cache.get_solved(mol.inchi_key)
                    if solved_data:
                        self.solved[mol.inchi_key] = solved_data
                    return cscore

        feasible_actions = self._determine_feasible_actions(mol)

        score = 0.0
        for action in feasible_actions:
            reactants = action.reactants
            score = sum([self.req_search_tree(x, depth + 1) for x in reactants[0]]) / len(reactants[0])
            if score > self.app_config.search.score_acceptance_threshold:
                self.solved[mol.inchi_key] = (reactants[0], score, action.metadata['classification'])
                self._update_cache(mol, depth, score)
                return score

        self._update_cache(mol, depth, score)
        return score

    def best_route(self, mol: TreeMolecule, depth: int, tree: defaultdict):
        # Past the depth limit a node is a route leaf — never expand it further,
        # even if it is in self.solved (that knowledge came from a shallower
        # search of the same molecule as its own target). Forcing tup=None here
        # routes such boundary nodes into the leaf branch so they get stock-
        # checked, warned, and recorded in BBs instead of being silently dropped.
        tup = None if depth > self.max_depth else self.solved.get(mol.inchi_key)
        if tup is None:
            if mol not in self.stock:
                self._logger.warning(
                    f"best_route: {mol.smiles} is a route leaf but not in stock — "
                    "route truncated (depth boundary or cache/solved invariant)")
            self.BBs.append(mol.smiles)
            return
        rxn, score, clas = tup
        reactants = '.'.join([m.smiles for m in rxn])
        tree[depth + 1].append([f'{mol.smiles} => {reactants}', clas])
        for x in rxn:
            self.best_route(x, depth + 1, tree)

    def _update(self, mol: TreeMolecule, smi: str, score: float, tree: defaultdict, rows: List) -> List:
        if score > self.app_config.search.score_acceptance_threshold:
            self.best_route(mol, 0, tree)
            self._save_to_redis()  # Persist to Redis if available and successful
        rows.append({'SMILES': smi, 'score': score, 'route': dict(tree), 'BBs': self.BBs})
        return rows

    def _update_cache(self, mol: TreeMolecule, depth: int, score: float):
        self.cache[mol.inchi_key] = (depth, score)
        if self.redis_cache:
            self.redis_cache.set_cache(mol.inchi_key, depth, score)


