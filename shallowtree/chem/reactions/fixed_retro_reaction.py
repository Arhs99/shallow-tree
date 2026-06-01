from __future__ import annotations

from typing import Optional, Tuple, List, Dict

from shallowtree.chem.molecules.tree_molecule import TreeMolecule
from shallowtree.chem.molecules.unique_molecule import UniqueMolecule
from shallowtree.chem.reactions.reaction_interface_mixin import _ReactionInterfaceMixin
from shallowtree.chem.reactions.smiles_based_retro_reaction import SmilesBasedRetroReaction


class FixedRetroReaction(_ReactionInterfaceMixin):
    """
    A retrosynthesis reaction that has the same interface as `RetroReaction`
    but it is fixed so it does not support SMARTS application or any creation of reactants.

    The reactants are set by using the `reactants` property.

    :ivar mol: the UniqueMolecule object that this reaction is applied to
    :ivar smiles: the SMILES representation of the RDKit reaction
    :ivar metadata: meta data associated with the reaction
    :ivar reactants: the reactants of this reaction

    :param mol: the molecule
    :param smiles: the SMILES of the reaction
    :param metadata: some meta data
    """

    def __init__(
            self,
            mol: UniqueMolecule,
            smiles: str = "",
            metadata: Optional[Dict] = None,
    ) -> None:
        self.mol = mol
        self.smiles = smiles
        self.metadata = metadata or {}
        self.reactants: Tuple[Tuple[UniqueMolecule, ...], ...] = ()

    def copy(self) -> "FixedRetroReaction":
        """
        Shallow copy of this instance.

        :return: the copy
        """
        new_reaction = FixedRetroReaction(self.mol, self.smiles, self.metadata)
        new_reaction.reactants = tuple(mol_list for mol_list in self.reactants)
        return new_reaction

    def to_smiles_based_retroreaction(self) -> SmilesBasedRetroReaction:
        """
        Convert a FixedRetroReaction to a SmilesBasedRetroReaction.

        :return: the SmilesBasedRetroReaction.
        """
        if self.metadata and "mapped_reaction_smiles" in self.metadata.keys():
            mapped_reaction_smiles = self.metadata["mapped_reaction_smiles"]
        else:
            mapped_reaction_smiles = self.reaction_smiles()

        mapped_reaction_smiles = mapped_reaction_smiles.split(">>")
        product = mapped_reaction_smiles[0]
        reactants = mapped_reaction_smiles[1]

        return SmilesBasedRetroReaction(
            mol=TreeMolecule(smiles=product, parent=None),
            mapped_prod_smiles=product,
            reactants_str=reactants,
        )

    def _products_getter(self) -> Tuple[UniqueMolecule, ...]:
        return self.reactants[0]

    def _reactants_getter(self) -> List[UniqueMolecule]:
        return [self.mol]
