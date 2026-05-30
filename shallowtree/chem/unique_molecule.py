from __future__ import annotations

from typing import Optional

from shallowtree.chem.molecule import Molecule
from rdkit.Chem import Mol



class UniqueMolecule(Molecule):
    """
    A special molecule with the hash set to the id of the object.
    Therefore no two instances of this class will be comparable.

    :param rd_mol: a RDKit mol object to encapsulate, defaults to None
    :param smiles: a SMILES to convert to a molecule object, defaults to None
    :param sanitize: if True, the molecule will be immediately sanitized, defaults to False
    :raises MoleculeException: if neither rd_mol or smiles is given, or if the molecule could not be sanitized
    """

    def __init__(
            self,
            rd_mol: Optional[Mol] = None,
            smiles: Optional[str] = None,
            sanitize: bool = False,
    ):
        super().__init__(rd_mol=rd_mol, smiles=smiles, sanitize=sanitize)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, _) -> bool:
        return False
