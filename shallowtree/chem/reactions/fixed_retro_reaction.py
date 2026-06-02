from __future__ import annotations

from typing import Optional, Tuple, List, Dict

import numpy as np

from shallowtree.chem.molecules.tree_molecule import TreeMolecule
from shallowtree.chem.molecules.unique_molecule import UniqueMolecule
from shallowtree.chem.reactions.smiles_based_retro_reaction import SmilesBasedRetroReaction


class FixedRetroReaction:
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

    def fingerprint(self, radius: int, nbits: Optional[int] = None, chiral: bool = False) -> np.ndarray:
        """
        Returns a difference fingerprint

        :param radius: the radius of the fingerprint
        :param nbits: the length of the fingerprint. If not given it will use RDKit default, defaults to None
        :param chiral: if True, include chirality information
        :return: the fingerprint
        """
        product_fp = sum(mol.fingerprint(radius, nbits, chiral) for mol in self._products_getter()) # type: ignore
        reactants_fp = sum(mol.fingerprint(radius, nbits, chiral) for mol in self._reactants_getter()) # type: ignore
        return reactants_fp - product_fp  # type: ignore

    def reaction_smiles(self) -> str:
        """
        Get the reaction SMILES, i.e. the SMILES of the reactants and products joined together

        :return: the SMILES
        """
        reactants = ".".join(mol.smiles for mol in self._reactants_getter())  # type: ignore
        products = ".".join(mol.smiles for mol in self._products_getter())  # type: ignore
        return f"{reactants}>>{products}"

    def _products_getter(self) -> Tuple[UniqueMolecule, ...]:
        return self.reactants[0]

    def _reactants_getter(self) -> List[UniqueMolecule]:
        return [self.mol]
