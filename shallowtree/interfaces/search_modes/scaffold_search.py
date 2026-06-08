from collections import defaultdict
from typing import List, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from shallowtree.chem.molecules.tree_molecule import TreeMolecule
from shallowtree.interfaces.search_modes.base_tree_search import BaseTreeSearch


class ScaffoldSearch(BaseTreeSearch):

    def search(self, smiles: List[str], max_depth=2) -> pd.DataFrame:
        scaffold_str = self._input_config.scaffold
        self.max_depth = max_depth
        rows = []
        context_scaffold = self._parse_scaffold_query(scaffold_str)
        # Scaffold-matching reactants are intentional terminal nodes here and
        # are never added to self.solved; best_route uses this to suppress its
        # invariant warning for them.
        wildcard_info = self._scaffold_wildcard_info(context_scaffold)
        # Used by _matches_context_scaffold to suppress best_route warnings
        # for relaxed-branch terminals (e.g. the phenol on the scaffold side
        # of a Williamson cut, which carries an OH where the wildcard sat
        # in the root and so doesn't match the strict scaffold).
        context_scaffold_stripped = None if wildcard_info is None else wildcard_info[1]

        for smi in smiles:
            solution = defaultdict(list)
            mol = TreeMolecule(parent=None, smiles=smi)
            self.BBs = []

            # Pre-populate from Redis if available
            self._load_from_redis(mol)
            feasible_actions = self._determine_feasible_actions(mol)
            score = self._solve_and_score_routes(mol, context_scaffold, feasible_actions, wildcard_info)
            rows = self._update(mol, smi, score, solution, rows, context_scaffold, context_scaffold_stripped)
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

    def best_route(self, mol: TreeMolecule, depth: int, tree: defaultdict, context_scaffold: Mol = None,
                   context_scaffold_stripped: Mol = None):
        # Past the depth limit a node is a route leaf — never expand it further,
        # even if it is in self.solved (that knowledge came from a shallower
        # search of the same molecule as its own target). Forcing tup=None here
        # routes such boundary nodes into the leaf branch so they get stock-
        # checked, warned, and recorded in BBs instead of being silently dropped.
        tup = None if depth > self.max_depth else self.solved.get(mol.inchi_key)
        if tup is None:
            if mol not in self.stock and not self._matches_context_scaffold(mol, context_scaffold, context_scaffold_stripped):
                self._logger.warning(
                    f"best_route: {mol.smiles} is a route leaf but not in stock — "
                    "route truncated (depth boundary or cache/solved invariant)")
            self.BBs.append(mol.smiles)
            return
        rxn, score, clas = tup
        reactants = '.'.join([m.smiles for m in rxn])
        tree[depth + 1].append([f'{mol.smiles} => {reactants}', clas])
        for x in rxn:
            self.best_route(x, depth + 1, tree, context_scaffold, context_scaffold_stripped)

    @staticmethod
    def _parse_scaffold_query(scaffold_str: str):
        # Try SMILES first so RDKit perceives aromaticity — lets Kekulé-form
        # scaffolds (e.g. "C1N=CSC=1...") match aromatic targets. Promote to a
        # query mol so dummy atoms ([*] / *) behave as wildcards. Fall back to
        # SMARTS for true SMARTS expressions like [#6;R], [!#1], recursive SMARTS.
        mol = Chem.MolFromSmiles(scaffold_str)
        if mol is not None:
            return Chem.AdjustQueryProperties(mol)
        return Chem.MolFromSmarts(scaffold_str)

    @staticmethod
    def _scaffold_wildcard_info(scaffold: Mol) -> Tuple:
        # If the scaffold has exactly one leaf wildcard ([*] / *, atomic
        # num 0, degree 1), return (wildcard_atom_idx, stripped_scaffold).
        # The stripped scaffold (wildcard atom removed) is what we match
        # against a reactant whose disconnection cut the wildcard bond and
        # produced an H-terminated end (e.g. Williamson retro: ArO[*] ->
        # ArOH + [*]-X). Returns None when the relaxed boundary check
        # should not apply (no wildcard, multiple wildcards, or internal
        # wildcard).
        wildcards = [i for i, a in enumerate(scaffold.GetAtoms()) if a.GetAtomicNum() == 0]
        if len(wildcards) != 1:
            return None
        wildcard_idx = wildcards[0]
        if scaffold.GetAtomWithIdx(wildcard_idx).GetDegree() != 1:
            return None
        rw = Chem.RWMol(scaffold)
        rw.RemoveAtom(wildcard_idx)
        return wildcard_idx, rw.GetMol()

    @staticmethod
    def _find_strict_boundary_match(reactants, scaffold, root_match):
        # Strict boundary check: a heavy-atom-for-heavy-atom swap at the
        # scaffold edge (Suzuki / Buchwald-Hartwig style).
        for r in reactants:
            r_match = set(r.index_to_mapping[x] for x in r.rd_mol.GetSubstructMatch(scaffold))
            if r_match and len(r_match ^ root_match) == 2:
                return r
        return None

    @staticmethod
    def _find_relaxed_boundary_match(reactants, scaffold_stripped, root_match, wildcard_mapping):
        # Relaxed boundary check: catches retros that produce an H-terminated
        # end at the wildcard position (Williamson ether, amide hydrolysis,
        # ...). Requires that the only atom missing from the reactant's
        # scaffold-minus-wildcard match is exactly the wildcard's mapping in
        # the root, AND that the wildcard's atom is gone from the reactant
        # entirely (not merely absent from this particular match) — otherwise
        # a disconnection elsewhere that leaves the scaffold intact would pass.
        if wildcard_mapping is None or scaffold_stripped is None:
            return None
        for r in reactants:
            if wildcard_mapping in r.index_to_mapping.values():
                continue
            for hit in r.rd_mol.GetSubstructMatches(scaffold_stripped):
                r_strip = set(
                    r.index_to_mapping[i] for i in hit if i in r.index_to_mapping
                )
                if r_strip and root_match - r_strip == {wildcard_mapping}:
                    return r
        return None

    def _create_wildcard_mapping(self, mol: TreeMolecule, wildcard_info, root_hit) -> Tuple:
        wildcard_mapping = None
        scaffold_stripped = None
        if wildcard_info is not None and root_hit:
            wildcard_idx, scaffold_stripped = wildcard_info
            w_atom_idx = root_hit[wildcard_idx]
            wildcard_mapping = mol.index_to_mapping.get(w_atom_idx)
        return wildcard_mapping, scaffold_stripped

    def _solve_and_score_routes(self, mol: TreeMolecule, scaffold, feasible_actions: List,
                                wildcard_info=None) -> float:
        score = 0
        root_hit = mol.rd_mol.GetSubstructMatch(scaffold)
        root_match = set(mol.index_to_mapping[x] for x in root_hit)
        wildcard_mapping, scaffold_stripped = self._create_wildcard_mapping(mol, wildcard_info, root_hit)

        for action in feasible_actions:
            reactants = action.reactants[0]
            strict = self._find_strict_boundary_match(reactants, scaffold, root_match)
            relaxed = (
                self._find_relaxed_boundary_match(reactants, scaffold_stripped, root_match, wildcard_mapping)
                if strict is None
                else None
            )
            if strict:
                chosen = strict
            elif relaxed :
                chosen = relaxed
            else:
                continue
            score_list = [self.req_search_tree(x, 1) for x in reactants if x != chosen]
            score = sum(score_list) / (len(reactants) - 1)
            if score > self.app_config.search.score_acceptance_threshold:
                self.solved[mol.inchi_key] = (reactants, score, action.metadata['classification'])
        return score

    def _update(self, mol: TreeMolecule, smi: str, score: float, tree: defaultdict, rows: List,
                context_scaffold: Mol = None, context_scaffold_stripped: Mol = None) -> List:
        if score > self.app_config.search.score_acceptance_threshold:
            self.best_route(mol, 0, tree, context_scaffold, context_scaffold_stripped)
            self._save_to_redis()  # Persist to Redis if available and successful
        rows.append({'SMILES': smi, 'score': score, 'route': dict(tree), 'BBs': self.BBs})
        return rows

    def _update_cache(self, mol: TreeMolecule, depth: int, score: float):
        self.cache[mol.inchi_key] = (depth, score)
        if self.redis_cache:
            self.redis_cache.set_cache(mol.inchi_key, depth, score)

    def _matches_context_scaffold(self, mol: TreeMolecule, context_scaffold: Mol, context_scaffold_stripped: Mol) -> bool:
        if context_scaffold is None:
            return False
        if mol.rd_mol.GetSubstructMatch(context_scaffold):
            return True
        # Relaxed-branch terminals carry an H where the wildcard sat (Williamson
        # phenol etc.) and don't match the strict scaffold — match the stripped
        # form instead so best_route doesn't warn on them.
        if context_scaffold_stripped:
            return bool(mol.rd_mol.GetSubstructMatch(context_scaffold_stripped))
        return False
