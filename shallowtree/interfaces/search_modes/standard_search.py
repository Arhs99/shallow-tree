
from __future__ import annotations

import time
from collections import defaultdict
from typing import List

import pandas as pd
from rdkit.Chem.rdchem import Mol

from shallowtree.chem.molecules.tree_molecule import TreeMolecule
from shallowtree.interfaces.search_modes.base_tree_search import BaseTreeSearch


class StandardSearch(BaseTreeSearch):

    def search(self, smiles: List[str]) -> pd.DataFrame:
        rows = []
        for smi in smiles:
            start_time = time.time()
            solution = defaultdict(list)
            mol = TreeMolecule(parent=None, smiles=smi)
            building_blocks = []

            self._load_from_redis(mol)
            try:
                score, resolved = self.req_search_tree(mol, depth=0, start_time=start_time)
                rows = self._update(mol, smi, score, resolved, solution, rows, building_blocks, start_time)
            except TimeoutError:
                rows.append(
                    {'SMILES': smi, 'score': 0, 'resolved': False, 'route': dict(solution), 'BBs': building_blocks,
                     'search_duration': 'Exceeded'})
            self.solved = {}
            self.cache = {}

        df = pd.DataFrame(rows)
        return df

    def best_route(self, mol: TreeMolecule, depth: int, tree: defaultdict, building_blocks: List,
                   ancestors: frozenset = frozenset()) -> bool:
        # Reconstructs the route and returns whether it is fully resolved — every
        # emitted leaf genuinely in stock. This re-validates the gate's verdict,
        # which is computed over the inchi_key cache and can be optimistic across
        # depths (a node resolved with spare budget may be truncated here when the
        # concrete path reaches it at the depth boundary).

        # Defensive cycle guard mirroring req_search_tree: never recurse into a
        # molecule already on the current route path — emit it as a (non-resolved)
        # leaf instead of looping forever.
        if mol.inchi_key in ancestors:
            building_blocks.append(mol.smiles)
            return False
        # Past the depth limit a node is a route leaf — never expand it further,
        # even if it is in self.solved (that knowledge came from a shallower
        # search of the same molecule as its own target). Forcing tup=None here
        # routes such boundary nodes into the leaf branch so they get stock-
        # checked, warned, and recorded in BBs instead of being silently dropped.
        tup = None if depth > self._input_config.depth else self.solved.get(mol.inchi_key)
        if tup is None:
            in_stock = mol in self.stock
            if not in_stock:
                self._logger.warning(
                    f"best_route: {mol.smiles} is a route leaf but not in stock — "
                    "route truncated (depth boundary or cache/solved invariant)")
            building_blocks.append(mol.smiles)
            return in_stock
        rxn, score, clas = tup
        reactants = '.'.join([m.smiles for m in rxn])
        tree[depth + 1].append([f'{mol.smiles} => {reactants}', clas])
        resolved = True
        for x in rxn:
            # recursive call first so the full route/BBs are always built (no short-circuit)
            resolved = self.best_route(x, depth + 1, tree, building_blocks, ancestors | {mol.inchi_key}) and resolved
        return resolved

    def _update(self, mol: TreeMolecule, smi: str, score: float, resolved: bool, tree: defaultdict, rows: List,
                building_blocks: List, start_time: float) -> List:
        # The gate drives the search toward resolved routes; re-validate against the
        # actually reconstructed route so the reported ``resolved`` is honest (True
        # only when every leaf is really in stock). The soft ``score`` is retained
        # as a ranking signal.
        if resolved:
            resolved = self.best_route(mol, 0, tree, building_blocks)
            self._save_to_redis(start_time)  # Persist to Redis if available and successful
        delta = time.time() - start_time
        rows.append({'SMILES': smi, 'score': score, 'resolved': resolved, 'route': dict(tree), 'BBs': building_blocks,
                     'search_duration': str(delta)})
        return rows

