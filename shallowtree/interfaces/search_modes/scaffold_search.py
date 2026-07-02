from collections import defaultdict
import time
from typing import List, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from shallowtree.chem.molecules.tree_molecule import TreeMolecule
from shallowtree.interfaces.search_modes.base_tree_search import BaseTreeSearch


class ScaffoldSearch(BaseTreeSearch):

    def search(self, smiles: List[str], clear: bool = True) -> pd.DataFrame:
        scaffold_str = self._input_config.scaffold
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
            start_time = time.time()
            solution = defaultdict(list)
            mol = TreeMolecule(parent=None, smiles=smi)
            building_blocks = []

            # Pre-populate from Redis if available
            self._load_from_redis(mol)
            feasible_actions = self._determine_feasible_actions(mol)
            try:
                score, resolved = self._solve_and_score_routes(mol, context_scaffold, feasible_actions, wildcard_info, start_time)
                rows = self._update(mol, smi, score, resolved, solution, rows, building_blocks, start_time, context_scaffold, context_scaffold_stripped)
            except TimeoutError:
                rows.append(
                    {'SMILES': smi, 'score': 0, 'resolved': False, 'route': dict(solution), 'BBs': building_blocks,
                     'search_duration': 'Exceeded'})
            if clear:
                self.solved = {}
                self.cache = {}

        df = pd.DataFrame(rows)
        return df

    def best_route(self, mol: TreeMolecule, depth: int, tree: defaultdict, building_blocks: List,
                   context_scaffold: Mol = None, context_scaffold_stripped: Mol = None,
                   ancestors: frozenset = frozenset()) -> bool:
        # Reconstructs the route and returns whether it is fully resolved. A leaf
        # is resolved if it is in stock OR it is the intended context-scaffold
        # terminal (the deliberate stopping point, never a buyable building block).
        # This re-validates the gate's verdict, which is cache-optimistic across
        # depths (see StandardSearch.best_route).

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
            matched = self._matches_context_scaffold(mol, context_scaffold, context_scaffold_stripped)
            in_stock = mol in self.stock
            if not in_stock and not matched:
                self._logger.warning(
                    f"best_route: {mol.smiles} is a route leaf but not in stock — "
                    "route truncated (depth boundary or cache/solved invariant)")
            building_blocks.append(mol.smiles)
            return in_stock or matched
        rxn, score, clas = tup
        reactants = '.'.join([m.smiles for m in rxn])
        tree[depth + 1].append([f'{mol.smiles} => {reactants}', clas])
        resolved = True
        for x in rxn:
            # recursive call first so the full route/BBs are always built (no short-circuit)
            resolved = self.best_route(x, depth + 1, tree, building_blocks, context_scaffold,
                                       context_scaffold_stripped, ancestors | {mol.inchi_key}) and resolved
        return resolved

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
                                wildcard_info, start_time) -> Tuple[float, bool]:
        score = 0
        resolved = False
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
            # The chosen scaffold reactant is the intended terminal — it is excluded
            # from both scoring and resolution (it carries the context scaffold, not a
            # stock building block). "Resolved" therefore means every OTHER reactant
            # bottoms out in stock.
            child_results = [self.req_search_tree(x, 1, start_time=start_time, ancestors=frozenset({mol.inchi_key})) for x in reactants if x != chosen]
            score = sum(s for s, _ in child_results) / (len(reactants) - 1)
            if all(g for _, g in child_results):
                # Last resolved disconnection wins (preserves the pre-existing
                # scaffold route selection); no early break.
                self.solved[mol.inchi_key] = (reactants, score, action.metadata['classification'])
                resolved = True
        return score, resolved

    def _update(self, mol: TreeMolecule, smi: str, score: float, resolved: bool, tree: defaultdict, rows: List, building_blocks: List,
                start_time: float, context_scaffold: Mol = None, context_scaffold_stripped: Mol = None) -> List:
        # The gate drives the search toward resolved routes; re-validate against the
        # actually reconstructed route so the reported ``resolved`` is honest (all
        # non-scaffold leaves really in stock). The soft ``score`` is retained as a
        # ranking signal.
        if resolved:
            resolved = self.best_route(mol, 0, tree, building_blocks, context_scaffold, context_scaffold_stripped)
            self._save_to_redis(start_time)  # Persist to Redis if available and successful
        delta = time.time() - start_time
        rows.append({'SMILES': smi, 'score': score, 'resolved': resolved, 'route': dict(tree), 'BBs': building_blocks,
                     'search_duration': str(delta)})
        return rows

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
