from __future__ import annotations

import time
import json
from collections import defaultdict
from typing import TYPE_CHECKING

from rdkit import Chem
import pandas as pd

from shallowtree.chem import Molecule, TreeMolecule
from shallowtree.context.config import Configuration

# This must be imported first to setup logging for rdkit, tensorflow etc
from shallowtree.utils.logging import logger

if TYPE_CHECKING:
    from shallowtree.chem import RetroReaction
    from shallowtree.utils.type_utils import (
        Callable,
        Dict,
        List,
        Optional,
        StrDict,
        Tuple,
        Union,
    )


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
        self.filter_policy = self.config.filter_policy
        self.stock = self.config.stock

        self.max_depth = 2

    def context_search(self, smiles: List[str], scaffold_str: str, max_depth=2) -> pd.DataFrame:
        self.max_depth = max_depth
        rows = []
        for smi in smiles:
            start = time.time()
            print(smi)
            solution = defaultdict(list)
            mol = TreeMolecule(parent=None, smiles=smi)
            scaffold = Chem.MolFromSmarts(scaffold_str)
            actions, _ = self.expansion_policy.get_actions([mol])
            self.solved = dict()
            self.cache = dict()
            self._counter = 0
            self._cache_counter = 0
            score = 0.0
            for action in actions:
                reactants = action.reactants
                feasibility_prob = 0
                if not reactants:
                    continue
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
            print(
                f'Score: {score}  in {time.time() - start} sec  {self._counter} evaluations  {self._cache_counter} cache hits')
            if score > 0.9:
                self.reconstruct_tree(mol, 0, solution)
                json_data = json.dumps(dict(solution), indent=2)
                print(json_data)
            print('*' * 20)
            rows.append({'SMILES': smi, 'score': score, 'route': dict(solution)})
        df = pd.DataFrame(rows)
        return df

    def search_tree(
            self,
            smiles: List[str],
            filter_func: Optional[Callable[[RetroReaction], bool]] = None,
            max_depth=2
    ) -> pd.DataFrame:
        """
        """
        self.max_depth = max_depth
        rows = []
        for smi in smiles:
            start = time.time()
            print(smi)
            solution = defaultdict(list)
            mol = TreeMolecule(parent=None, smiles=smi)
            if mol in self.stock:
                print('in stock!')
            self.solved = dict()
            self.cache = dict()
            self._counter = 0
            self._cache_counter = 0
            score = self.req_search_tree(mol, depth=0)
            print(
                f'Score: {score}  in {time.time() - start} sec  {self._counter} evaluations  {self._cache_counter} cache hits')
            if score > 0.9:
                self.reconstruct_tree(mol, 0, solution)
                json_data = json.dumps(dict(solution), indent=2)
                print(json_data)

            print('*' * 20)
            rows.append({'SMILES': smi, 'score': score, 'route': dict(solution)})
        df = pd.DataFrame(rows)
        return df

    def req_search_tree(self, mol: TreeMolecule, depth: int) -> float:
        if depth > self.max_depth:
            return 0.0
        self._counter += 1
        if mol in self.stock:
            # print(f'depth: {depth}  molecule in stock: {mol.smiles}')
            return 1.0
        if mol.inchi_key in self.cache.keys():
            self._cache_counter += 1
            cdepth, cscore = self.cache[mol.inchi_key]
            if cdepth <= depth:
                return cscore

        actions, _ = self.expansion_policy.get_actions([mol])
        score = 0.0
        for action in actions:
            reactants = action.reactants
            feasibility_prob = 0
            if not reactants:
                continue
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
                # A = [x.smiles for x in reactants[0]]
                # print(f'depth: {depth}  {mol.smiles}>>{A}')
                return score
        self.cache[mol.inchi_key] = (depth, score)
        return score

    def reconstruct_tree(self, mol: Molecule, depth: int, tree: defaultdict):
        while depth <= self.max_depth:
            tup = self.solved.get(mol.inchi_key)
            if tup is None:
                return
            else:
                rxn, score, clas = tup
                reactants = '.'.join([m.smiles for m in rxn])
                tree[depth + 1].append([f'{mol.smiles} => {reactants}', clas])
                for x in rxn:
                    self.reconstruct_tree(x, depth + 1, tree)
                return


if __name__ == '__main__':
    smiles = [
        "CN(C)CC(O)COc1ccc(Nc2nccc(N(Cc3ccccc3)c3cc(Cl)ccc3Cl)n2)cc1",
        "O=C(Nc1n[nH]c2cc(-c3ccccc3)ccc12)C1CC1",
        "O=C(Nc1n[nH]c2nc(-c3ccc(O)c(Br)c3)ccc12)C1CC1",
        "O=C(Nc1ccc(N2CCNCC2)cc1)c1n[nH]cc1Nc1cc(Oc2ccccc2)ncn1",
        "CN(C)CC(O)COc1ccc(Nc2cc(Nc3ccccc3F)ncn2)cc1",
        "CC(C)n1cnc2c(N)nc(NCCCO)nc21",
        "Cc1ccc2[nH]cc(C(=O)C(=O)N3CCc4ccccc43)c2c1",
        "CCSc1nn2c(Cc3ccccc3)nnc2s1",
        "CCc1cc2c(N/N=C\c3cccs3)nc(-c3ccccc3)nc2s1",
        "NCC(=O)NS(=O)(=O)c1ccc(Nc2nc(N)n(C(=O)c3c(F)cccc3F)n2)cc1",
        "CC1(C)OCC(=O)Nc2cc(Nc3nc(NCC(F)(F)F)c4occc4n3)ccc21",
        "NS(=O)(=O)c1ccc(N/N=C2\C(=O)Nc3cc(Br)ccc32)cc1",
        "O=C1Nc2ccc(F)c3c(OCCCO)cc(-c4ccc[nH]4)c1c23",
        "COc1cc(C)c(Sc2cnc(NC(=O)c3ccc(CNC(C)C(C)(C)C)cc3)s2)cc1C(=O)N1CCN(C(C)=O)CC1",
        "Cc1ccc(Nc2cc(Nc3ccc(OCC(O)CN(C)C)cc3)ncn2)cc1",
        "CC(C)(C)c1ccc(Nc2nccc(-c3cnn4ncccc34)n2)cc1",
        "COc1cc(O)c2c(c1)C(=O)c1cc(C)c(O)c(O)c1C2=O",
        "CC(C)n1cnc2c(NC3CCCC3)nc(NCCCO)nc21",
        "O=C(Nc1n[nH]c2cc(-c3ccc(F)cc3)ccc12)C1CC1",
        "O=C(Nc1n[nH]c2nc(-c3ccco3)c(Br)cc12)C1CCN(Cc2ccccc2)C1",
        "CC(=O)Nc1c[nH]nc1-c1nc2ccccc2[nH]1",
        "O=C(Nc1c[nH]nc1-c1nc2ccc(CN3CCOCC3)cc2[nH]1)c1ccc(F)cc1",
        "O=C(Nc1c[nH]nc1-c1nc2cc(CN3CCOCC3)ccc2[nH]1)Nc1c(F)cccc1F",
        "CC1NCCCC1C(=O)Nc1ncc(SCc2ncc(C(C)(C)C)o2)s1",
        "O=C(Nc1n[nH]c2cc(-c3cccs3)ccc12)C1CC1",
        "CC(C)(C)c1cnc(CSc2cnc(NCC3CCNCC3)s2)o1",
        "Cc1ccc(C(=O)NC2CC2)cc1-c1ccc2c(C3CCNCC3)noc2c1",
        "CN(C)c1cc2c(Nc3ccc4c(cnn4Cc4ccccc4)c3)ncnc2cn1",
        "Nc1ccccc1NC(=O)c1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
        "N#Cc1c(NC(=O)c2cccc3ccccc23)sc2c1CCCC2",
        "NS(=O)(=O)c1ccc(N/N=C2\C(=O)Nc3ccc4ncsc4c32)cc1",
        "NS(=O)(=O)c1ccc(N/N=C2\C(=O)Nc3cc(Oc4ccccc4)ccc32)cc1",
        "CNC(=O)c1nn(C)c2c1C(C)(C)Cc1cnc(Nc3ccc(CN4CCN(C)CC4)cc3)nc1-2",
        "CC(C)(C)c1cc2c(N/N=C\c3cccc(CN)n3)ncnc2s1",
        "COc1cccc2c1c(Cl)c1c3c(cc(O)c(O)c32)C(=O)N1",
    ]
    filename = '/home/kostas/data/aizynth/config.yml'
    start = time.time()
    expander = Expander(configfile=filename)
    expander.expansion_policy.select_first()
    print(f'Load exp policy: {time.time() - start} sec')
    expander.filter_policy.select_first()
    print(f'Load filter policy: {time.time() - start} sec')
    expander.stock.select_first()
    print(f'Load stock: {time.time() - start} sec')

    df = expander.context_search(
        ['Clc1ccccc1COC5CC(Nc3n[nH]c4cc(c2ccccc2)ccc34)C5'], scaffold_str='[*]c1n[nH]c2cc(-c3ccccc3)ccc12'
    )

    # df = expander.search_tree(
    #     # smiles[:5]
    #     ['C(Nc1n[nH]c2cc(-c3ccccc3)ccc12)C1CC1']
    #     # 'Cc1cccc(c1N(CC(=O)Nc2ccc(cc2)c3ncon3)C(=O)C4CCS(=O)(=O)CC4)C'
    # )
    print(df.to_csv())
