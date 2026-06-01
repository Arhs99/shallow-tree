from __future__ import annotations

import hashlib
from typing import Optional, List

import numpy as np
from rdkit.Chem import AllChem

from shallowtree.utils.type_utils import RdReaction


class _ReactionInterfaceMixin:
    """
    Mixin class to define a common interface for all reaction class

    The methods `_products_getter` and `_reactants_getter` needs to be implemented by subclasses
    """

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

    def hash_list(self) -> List[str]:
        """
        Return all the products and reactants as hashed SMILES

        :return: the hashes of the SMILES string
        """
        mols = self.reaction_smiles().replace(".", ">>").split(">>")
        return [hashlib.sha224(mol.encode("utf8")).hexdigest() for mol in mols]

    def hash_key(self) -> str:
        """
        Return a code that can be use to identify the reaction

        :return: the hash code
        """
        reactants = sorted([mol.inchi_key for mol in self._reactants_getter()])  # type: ignore
        products = sorted([mol.inchi_key for mol in self._products_getter()])  # type: ignore
        hash_ = hashlib.sha224()
        for item in reactants + [">>"] + products:
            hash_.update(item.encode())
        return hash_.hexdigest()

    def rd_reaction_from_smiles(self) -> RdReaction:
        """
        The reaction as a RDkit reaction object but created from the reaction smiles
        instead of the SMARTS of the template.

        :return: the reaction object
        """
        return AllChem.ReactionFromSmarts(self.reaction_smiles(), useSmiles=True)

    def reaction_smiles(self) -> str:
        """
        Get the reaction SMILES, i.e. the SMILES of the reactants and products joined together

        :return: the SMILES
        """
        reactants = ".".join(mol.smiles for mol in self._reactants_getter())  # type: ignore
        products = ".".join(mol.smiles for mol in self._products_getter())  # type: ignore
        return f"{reactants}>>{products}"
