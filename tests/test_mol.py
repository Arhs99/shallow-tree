"""Tests for shallowtree.chem.mol — Molecule, TreeMolecule, UniqueMolecule."""
import unittest

import numpy as np
from rdkit import Chem

from shallowtree.chem.mol import Molecule, TreeMolecule, UniqueMolecule
from shallowtree.utils.exceptions import MoleculeException


class TestMoleculeInit(unittest.TestCase):
    """Test Molecule construction."""

    def test_create_from_smiles(self):
        mol = Molecule(smiles="CCO")
        self.assertEqual(mol.smiles, "CCO")
        self.assertIsNotNone(mol.rd_mol)

    def test_create_from_rdmol(self):
        rd = Chem.MolFromSmiles("CCO")
        mol = Molecule(rd_mol=rd)
        self.assertIsNotNone(mol.smiles)
        self.assertIsNotNone(mol.rd_mol)

    def test_raises_on_empty_input(self):
        with self.assertRaises(MoleculeException):
            Molecule()

    def test_sanitize_flag(self):
        mol = Molecule(smiles="CCO", sanitize=True)
        self.assertTrue(mol._is_sanitized)

    def test_not_sanitized_by_default(self):
        mol = Molecule(smiles="CCO")
        self.assertFalse(mol._is_sanitized)


class TestMoleculeInChIKey(unittest.TestCase):
    """Test InChI key generation."""

    def test_deterministic(self):
        mol1 = Molecule(smiles="CCO")
        mol2 = Molecule(smiles="CCO")
        self.assertEqual(mol1.inchi_key, mol2.inchi_key)

    def test_same_molecule_different_smiles(self):
        mol1 = Molecule(smiles="CCO")
        mol2 = Molecule(smiles="OCC")
        self.assertEqual(mol1.inchi_key, mol2.inchi_key)

    def test_different_molecules_differ(self):
        mol1 = Molecule(smiles="CCO")
        mol2 = Molecule(smiles="CCCO")
        self.assertNotEqual(mol1.inchi_key, mol2.inchi_key)

    def test_aromatic_vs_kekulized_benzene(self):
        mol1 = Molecule(smiles="c1ccccc1")
        mol2 = Molecule(smiles="C1=CC=CC=C1")
        self.assertEqual(mol1.inchi_key, mol2.inchi_key)


class TestMoleculeEqualityHash(unittest.TestCase):
    """Test __eq__ and __hash__ via InChI key."""

    def test_equal_same_smiles(self):
        mol1 = Molecule(smiles="CCO")
        mol2 = Molecule(smiles="CCO")
        self.assertEqual(mol1, mol2)

    def test_equal_different_smiles_same_molecule(self):
        mol1 = Molecule(smiles="CCO")
        mol2 = Molecule(smiles="OCC")
        self.assertEqual(mol1, mol2)

    def test_hash_equal_for_same_molecule(self):
        mol1 = Molecule(smiles="CCO")
        mol2 = Molecule(smiles="OCC")
        self.assertEqual(hash(mol1), hash(mol2))

    def test_molecules_work_in_set(self):
        mol1 = Molecule(smiles="CCO")
        mol2 = Molecule(smiles="OCC")
        mol3 = Molecule(smiles="CCCO")
        s = {mol1, mol2, mol3}
        self.assertEqual(len(s), 2)

    def test_not_equal_to_non_molecule(self):
        mol = Molecule(smiles="CCO")
        self.assertNotEqual(mol, "CCO")
        self.assertNotEqual(mol, 42)


class TestBasicCompare(unittest.TestCase):
    """Test basic_compare (ignores stereochemistry)."""

    def test_ignores_stereochemistry(self):
        # L-alanine vs D-alanine
        mol1 = Molecule(smiles="N[C@@H](C)C(=O)O")
        mol2 = Molecule(smiles="N[C@H](C)C(=O)O")
        self.assertTrue(mol1.basic_compare(mol2))

    def test_different_molecules_dont_match(self):
        mol1 = Molecule(smiles="CCO")
        mol2 = Molecule(smiles="CCCO")
        self.assertFalse(mol1.basic_compare(mol2))


class TestFingerprint(unittest.TestCase):
    """Test Morgan fingerprint generation."""

    def setUp(self):
        self.mol = Molecule(smiles="CCO", sanitize=True)

    def test_shape(self):
        fp = self.mol.fingerprint(radius=2, nbits=2048)
        self.assertEqual(fp.shape, (2048,))

    def test_deterministic(self):
        fp1 = self.mol.fingerprint(radius=2, nbits=2048)
        fp2 = self.mol.fingerprint(radius=2, nbits=2048)
        np.testing.assert_array_equal(fp1, fp2)

    def test_cached_by_radius_nbits(self):
        fp1 = self.mol.fingerprint(radius=2, nbits=2048)
        fp2 = self.mol.fingerprint(radius=2, nbits=2048)
        self.assertIs(fp1, fp2)

    def test_binary_values(self):
        fp = self.mol.fingerprint(radius=2, nbits=2048)
        unique_vals = set(np.unique(fp))
        self.assertTrue(unique_vals.issubset({0.0, 1.0}))

    def test_different_radius_different_cache(self):
        fp1 = self.mol.fingerprint(radius=2, nbits=2048)
        fp2 = self.mol.fingerprint(radius=3, nbits=2048)
        self.assertIsNot(fp1, fp2)


class TestSanitize(unittest.TestCase):
    """Test sanitize method."""

    def test_idempotent(self):
        mol = Molecule(smiles="CCO")
        mol.sanitize()
        smiles1 = mol.smiles
        mol.sanitize()
        smiles2 = mol.smiles
        self.assertEqual(smiles1, smiles2)

    def test_clears_cache(self):
        mol = Molecule(smiles="CCO")
        _ = mol.inchi_key  # populate cache
        mol._is_sanitized = False  # force re-sanitize
        mol.sanitize()
        # After sanitize, cache is cleared and rebuilt on access
        self.assertIsNotNone(mol.inchi_key)

    def test_raises_on_invalid_molecule(self):
        mol = Molecule(smiles="CCO")
        # Corrupt the rd_mol to force sanitize failure
        mol.rd_mol = Chem.MolFromSmiles("[InvalidMol", sanitize=False)
        mol._is_sanitized = False
        with self.assertRaises(MoleculeException):
            mol.sanitize(raise_exception=True)


class TestTreeMolecule(unittest.TestCase):
    """Test TreeMolecule specifics."""

    def test_root_assigns_atom_mappings(self):
        mol = TreeMolecule(parent=None, smiles="CCO")
        for atom in mol.mapped_mol.GetAtoms():
            self.assertGreater(atom.GetAtomMapNum(), 0)

    def test_child_increments_transform(self):
        parent = TreeMolecule(parent=None, smiles="CCO")
        child = TreeMolecule(parent=parent, smiles="CC")
        self.assertEqual(child.transform, parent.transform + 1)

    def test_child_removes_atom_mapping_from_rd_mol(self):
        parent = TreeMolecule(parent=None, smiles="CCO")
        child = TreeMolecule(parent=parent, smiles="CC")
        for atom in child.rd_mol.GetAtoms():
            self.assertEqual(atom.GetAtomMapNum(), 0)

    def test_mapped_smiles_contains_mappings(self):
        mol = TreeMolecule(parent=None, smiles="CCO")
        # Mapped SMILES should contain atom map numbers like :1
        self.assertIn(":", mol.mapped_smiles)

    def test_root_transform_is_zero(self):
        mol = TreeMolecule(parent=None, smiles="CCO")
        self.assertEqual(mol.transform, 0)


class TestUniqueMolecule(unittest.TestCase):
    """Test UniqueMolecule hash/equality."""

    def test_hash_is_id(self):
        mol = UniqueMolecule(smiles="CCO")
        self.assertEqual(hash(mol), id(mol))

    def test_never_equal_even_same_smiles(self):
        mol1 = UniqueMolecule(smiles="CCO")
        mol2 = UniqueMolecule(smiles="CCO")
        self.assertNotEqual(mol1, mol2)

    def test_both_live_in_set(self):
        mol1 = UniqueMolecule(smiles="CCO")
        mol2 = UniqueMolecule(smiles="CCO")
        s = {mol1, mol2}
        self.assertEqual(len(s), 2)


if __name__ == "__main__":
    unittest.main()
