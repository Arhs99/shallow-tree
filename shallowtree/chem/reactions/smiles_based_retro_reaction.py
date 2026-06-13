from __future__ import annotations

from functools import partial
from typing import Optional, Any, Tuple, Set

from shallowtree.chem.molecules.molecule import Molecule
from shallowtree.chem.molecules.tree_molecule import TreeMolecule
from shallowtree.chem.reactions.retro_reaction import RetroReaction
from shallowtree.utils.exceptions import MoleculeException
from shallowtree.utils.type_utils import StrDict


class SmilesBasedRetroReaction(RetroReaction):
    """
    A retrosynthesis reaction where the SMILES of the reactants are given on initiation

    The SMILES representation of the reaction is the reaction SMILES

    :param mol: the molecule
    :param index: the index, defaults to 0
    :param metadata: some meta data
    :param reactants_str: a dot-separated string of reactant SMILES strings
    """

    _required_kwargs = ["reactants_str"]

    def __init__(
            self,
            mol: TreeMolecule,
            index: int = 0,
            metadata: Optional[StrDict] = None,
            **kwargs: Any,
    ):
        super().__init__(mol, index, metadata, **kwargs)
        self.reactants_str: str = kwargs["reactants_str"]
        self._mapped_prod_smiles = kwargs.get("mapped_prod_smiles")

    def __str__(self) -> str:
        return (
            f"retro reaction on molecule {self.mol.smiles} giving {self.reactants_str}"
        )

    def _apply(self) -> Tuple[Tuple[TreeMolecule, ...], ...]:
        outcomes = []
        smiles_list = self.reactants_str.split(".")

        exclude_nums = set(self.mol.mapping_to_index.keys())
        update_func = partial(self._remap, exclude_nums=exclude_nums)
        try:
            rct = tuple(
                TreeMolecule(
                    parent=self.mol,
                    smiles=smi,
                    sanitize=True,
                    mapping_update_callback=update_func,
                )
                for smi in smiles_list
            )
        except MoleculeException:
            pass
        else:
            outcomes.append(rct)
        self._reactants = tuple(outcomes)

        return self._reactants

    def _remap(self, mol: TreeMolecule, exclude_nums: Set[int]) -> None:
        """Find the mapping between parent and child and then re-map the child molecule"""
        if not self._mapped_prod_smiles:
            self._update_unmapped_atom_num(mol, exclude_nums)
            return

        parent_remapping = {}
        pmol = Molecule(smiles=self._mapped_prod_smiles, sanitize=True)
        for atom_idx1, atom_idx2 in enumerate(
                pmol.rd_mol.GetSubstructMatch(self.mol.mapped_mol)
        ):
            atom1 = self.mol.mapped_mol.GetAtomWithIdx(atom_idx1)
            atom2 = pmol.rd_mol.GetAtomWithIdx(atom_idx2)
            if atom1.GetAtomMapNum() > 0 and atom2.GetAtomMapNum() > 0:
                parent_remapping[atom2.GetAtomMapNum()] = atom1.GetAtomMapNum()

        for atom in mol.mapped_mol.GetAtoms():
            if atom.GetAtomMapNum() and atom.GetAtomMapNum() in parent_remapping:
                atom.SetAtomMapNum(parent_remapping[atom.GetAtomMapNum()])
            else:
                atom.SetAtomMapNum(0)

        self._update_unmapped_atom_num(mol, exclude_nums)

    def _make_smiles(self):
        rstr = ".".join(reactant.smiles for reactant in self.reactants[0])
        return f"{self.mol.smiles}>>{rstr}"
