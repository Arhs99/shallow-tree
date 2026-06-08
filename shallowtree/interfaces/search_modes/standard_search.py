
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

