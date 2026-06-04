from __future__ import annotations

import abc
from typing import List, Optional, Any, Tuple, Set, Dict

import numpy as np

from shallowtree.chem.molecules.tree_molecule import TreeMolecule


class RetroReaction(abc.ABC):
    """
    A retrosynthesis reaction. Only a single molecule is the reactant.

    This is an abstract class and child classes needs to implement the `_apply` and `_make_smiles` functions
    that should create the reactants molecule objects and the reaction SMILES representation, respectively.

    :ivar mol: the TreeMolecule object that this reaction is applied to
    :ivar index: a unique index of this reaction,
                 to count for the fact that a reaction can produce more than one outcome
    :ivar metadata: meta data associated with the reaction

    :param mol: the molecule
    :param index: the index, defaults to 0
    :param metadata: some meta data
    :params kwargs: any extra parameters for child classes
    """

    _required_kwargs: List[str] = []

    def __init__(
            self,
            mol: TreeMolecule,
            index: int = 0,
            metadata: Optional[Dict] = None,
            intern_cache: Optional[Dict] = None,
            **kwargs: Any,
    ) -> None:
        if any(name not in kwargs for name in self._required_kwargs):
            raise KeyError(
                f"A {self.__class__.__name__} class needs to be initiated "
                f"with keyword arguments: {', '.join(self._required_kwargs)}"
            )
        self.mol = mol
        self.index = index
        self.metadata: Dict = metadata or {}
        self._reactants: Optional[Tuple[Tuple[TreeMolecule, ...], ...]] = None
        self._smiles: Optional[str] = None
        self.intern_cache = intern_cache
        self._kwargs: Dict = kwargs

    def __str__(self) -> str:
        return f"reaction on molecule {self.mol.smiles}"

    @property
    def reactants(self) -> Tuple[Tuple[TreeMolecule, ...], ...]:
        """
        Returns the reactant molecules.
        Apply the reaction if necessary.

        :return: the products of the reaction
        """
        if not self._reactants:
            self._reactants = self._apply()
        return self._reactants

    @property
    def smiles(self) -> str:
        """
        The reaction as a SMILES

        :return: the SMILES
        """
        if self._smiles is None:
            try:
                self._smiles = self._make_smiles()
            except ValueError:
                self._smiles = ""  # noqa
        return self._smiles

    @property
    def unqueried(self) -> bool:
        """
        Return True if the reactants has never been retrieved
        """
        return self._reactants is None

    def mapped_reaction_smiles(self) -> str:
        """
        Get the mapped reaction SMILES if it exists
        :return: the SMILES
        """
        reactants = self.mol.mapped_smiles
        products = ".".join(mol.mapped_smiles for mol in self._products_getter())
        return reactants + ">>" + products

    @abc.abstractmethod
    def _apply(self) -> Tuple[Tuple[TreeMolecule, ...], ...]:
        pass

    @abc.abstractmethod
    def _make_smiles(self) -> str:
        pass

    def _products_getter(self) -> Tuple[TreeMolecule, ...]:
        return self.reactants[self.index]

    def _reactants_getter(self) -> List[TreeMolecule]:
        return [self.mol]

    @staticmethod
    def _update_unmapped_atom_num(mol: TreeMolecule, exclude_nums: Set[int]) -> None:
        mapped_nums = {num for num in mol.mapping_to_index.keys() if 0 < num < 900}
        offset = max(mapped_nums) + 1 if mapped_nums else 1
        for atom in mol.mapped_mol.GetAtoms():
            if 0 < atom.GetAtomMapNum() < 900:
                continue
            while offset in exclude_nums:
                offset += 1
            atom.SetAtomMapNum(offset)
            exclude_nums.add(offset)

    def reaction_smiles(self) -> str:
        """
        Get the reaction SMILES, i.e. the SMILES of the reactants and products joined together

        :return: the SMILES
        """
        reactants = ".".join(mol.smiles for mol in self._reactants_getter())  # type: ignore
        products = ".".join(mol.smiles for mol in self._products_getter())  # type: ignore
        return f"{reactants}>>{products}"

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
