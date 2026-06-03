from typing import Optional, Callable, Dict, List, Tuple, Sequence

from rdkit.Chem import Mol

from shallowtree.chem.molecules.molecule import Molecule
from shallowtree.utils.bonds import sort_bonds


class TreeMolecule(Molecule):
    """
    A special molecule that keeps a reference to a parent molecule.

    If the class is instantiated without specifying the `transform` argument,
    it is computed by increasing the value of the `parent.transform` variable.

    If no parent is provided the atoms with atom mapping number are tracked
    and inherited to children.

    :ivar mapped_mol: the tracked molecule with atom mappings
    :ivar mapped_smiles: the SMILES of the tracked molecule with atom mappings
    :ivar original_smiles: the SMILES as passed when instantiating the class
    :ivar parent: parent molecule
    :ivar transform: a numerical number corresponding to the depth in the tree

    :param parent: a TreeMolecule object that is the parent
    :param transform: the transform value, defaults to None
    :param rd_mol: a RDKit mol object to encapsulate, defaults to None
    :param smiles: a SMILES to convert to a molecule object, defaults to None
    :param sanitize: if True, the molecule will be immediately sanitized, defaults to False
    :param mapping_update_callback: if given will call this method before setting up the `mapped_smiles`
    :raises MoleculeException: if neither rd_mol or smiles is given, or if the molecule could not be sanitized
    """

    # pylint: disable=too-many-arguments
    def __init__(
            self,
            parent: Optional["TreeMolecule"],
            transform: Optional[int] = None,
            rd_mol: Optional[Mol] = None,
            smiles: Optional[str] = None,
            sanitize: bool = False,
            mapping_update_callback: Optional[Callable[["TreeMolecule"], None]] = None,
            intern_cache: Optional[Dict[str, "TreeMolecule"]] = None,
    ) -> None:
        super().__init__(parent=parent, transform=transform, rd_mol=rd_mol, smiles=smiles, sanitize=sanitize,
                         mapping_update_callback=mapping_update_callback, intern_cache=intern_cache)


    @property
    def mapping_to_index(self) -> Dict[int, int]:
        """Return a dictionary mapping to atom mappings to atom indices"""
        if not self._atom_mappings:
            self._atom_mappings = {
                atom.GetAtomMapNum(): atom.GetIdx()
                for atom in self.mapped_mol.GetAtoms()
                if atom.GetAtomMapNum()
            }
        return self._atom_mappings

    @property
    def mapped_atom_bonds(self) -> List[Tuple[int, int]]: #TODO: see if this is called often
        """Return a list of atom bonds as tuples on the mapped atom indices"""
        bonds = []
        for bond in self.mapped_mol.GetBonds():
            bonds.append((bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()))

        _atom_bonds = [
            (self.index_to_mapping[atom_index1], self.index_to_mapping[atom_index2])
            for atom_index1, atom_index2 in bonds
        ]
        return _atom_bonds

    def get_bonds_in_molecule(self, query_bonds: Sequence[Sequence[int]]) -> Sequence[Sequence[int]]:
        """
        Get bonds (from a list of bonds) that are present in the molecule.
        :param bonds: List of bond (atom pairs)
        :return: A list of bonds
        """
        molecule_bonds = sort_bonds(self.mapped_atom_bonds)
        query_bonds = sort_bonds(query_bonds)
        bonds_in_mol = [bond for bond in query_bonds if bond in molecule_bonds]
        return bonds_in_mol

    def has_all_focussed_bonds(self, bonds: Sequence[Sequence[int]]) -> bool:
        """Checks that the focussed bonds exist in the target molecule's atom bonds.

        :param bonds: Focussed bonds.
        :param target_mol: The target molecule.

        :return: A boolean indicating if the input bonds exist in the target molecule.
        """
        bonds_in_mol = self.get_bonds_in_molecule(bonds)
        return len(bonds_in_mol) == len(bonds)

