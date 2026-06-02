from __future__ import annotations

from rdchiral import main as rdc
from rdkit import Chem
from rdkit.Chem.rdchem import BondDir, BondStereo, ChiralType

from shallowtree.chem.molecules.tree_molecule import TreeMolecule
from shallowtree.utils.logging import logger

try:
    from rdchiral.bonds import get_atoms_across_double_bonds
    from rdchiral.initialization import BondDirOpposite
except ImportError:
    RDCHIRAL_CPP = True
else:
    RDCHIRAL_CPP = False
if RDCHIRAL_CPP:
    logger().warning(
        "WARNING: C++ version of RDChiral is supported, but with limited functionality"
    )

class _RdChiralProductWrapper:
    """
    Reimplementation of `rdchiralReaction`
    to preserve product molecule already created
    """

    # pylint: disable=W0106,C0103
    def __init__(self, product: TreeMolecule) -> None:
        product.sanitize()
        self.reactant_smiles = product.smiles

        # Initialize into RDKit mol
        self.reactants = Chem.Mol(product.mapped_mol.ToBinary())
        Chem.AssignStereochemistry(self.reactants, flagPossibleStereoCenters=True)
        self.reactants.UpdatePropertyCache(strict=False)

        self.atoms_r = {a.GetAtomMapNum(): a for a in self.reactants.GetAtoms()}
        self.idx_to_mapnum = lambda idx: self.reactants.GetAtomWithIdx(
            idx
        ).GetAtomMapNum()

        # Create copy of molecule without chiral information, used with
        # RDKit's naive runReactants
        self.reactants_achiral = Chem.Mol(product.rd_mol.ToBinary())
        [
            a.SetChiralTag(ChiralType.CHI_UNSPECIFIED)
            for a in self.reactants_achiral.GetAtoms()
        ]
        [
            (b.SetStereo(BondStereo.STEREONONE), b.SetBondDir(BondDir.NONE))
            for b in self.reactants_achiral.GetBonds()
        ]

        # Pre-list reactant bonds (for stitching broken products)
        self.bonds_by_mapnum = [
            (b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum(), b)
            for b in self.reactants.GetBonds()
        ]

        # Pre-list chiral double bonds (for copying back into outcomes/matching)
        self.bond_dirs_by_mapnum = {}
        for i, j, b in self.bonds_by_mapnum:
            if b.GetBondDir() != BondDir.NONE:
                self.bond_dirs_by_mapnum[(i, j)] = b.GetBondDir()
                self.bond_dirs_by_mapnum[(j, i)] = BondDirOpposite[b.GetBondDir()]

        # Get atoms across double bonds defined by mapnum
        self.atoms_across_double_bonds = get_atoms_across_double_bonds(self.reactants)

if RDCHIRAL_CPP:
    def _wrapper(mol):
        return rdc.rdchiralReactants(mol.mapped_smiles)


    _RdChiralProductWrapper = _wrapper  # type: ignore

_rdchiral_product_cache = {}
