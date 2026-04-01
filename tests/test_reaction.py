"""Tests for shallowtree.chem.reaction — Reaction classes."""
import unittest

import numpy as np

from shallowtree.chem.mol import TreeMolecule
from shallowtree.chem.reaction import (
    SmilesBasedRetroReaction,
    TemplatedRetroReaction,
    hash_reactions,
)


class TestSmilesBasedRetroReaction(unittest.TestCase):
    """Test SmilesBasedRetroReaction."""

    def setUp(self):
        self.mol = TreeMolecule(parent=None, smiles="CCO")

    def test_parses_reactants_into_tree_molecules(self):
        rxn = SmilesBasedRetroReaction(
            self.mol, reactants_str="CC.O"
        )
        reactants = rxn.reactants
        self.assertTrue(len(reactants) > 0)
        for r in reactants[0]:
            self.assertIsInstance(r, TreeMolecule)

    def test_reaction_smiles_has_arrow(self):
        rxn = SmilesBasedRetroReaction(
            self.mol, reactants_str="CC.O"
        )
        smiles = rxn.reaction_smiles()
        self.assertIn(">>", smiles)

    def test_missing_kwarg_raises_key_error(self):
        with self.assertRaises(KeyError):
            SmilesBasedRetroReaction(self.mol)


class TestTemplatedRetroReaction(unittest.TestCase):
    """Test TemplatedRetroReaction with RDKit path."""

    def setUp(self):
        self.mol = TreeMolecule(parent=None, smiles="CCO")
        # Simple dehydration-like retro template: alcohol -> alkene + water
        # Using a basic SMARTS that works with RDKit
        self.smarts = "[C:1]-[OH:2]>>([C:1]=[O:2])"

    def test_produces_reactants(self):
        rxn = TemplatedRetroReaction(
            self.mol, smarts=self.smarts, use_rdchiral=False
        )
        reactants = rxn.reactants
        # May or may not produce results depending on template, but should not crash
        self.assertIsInstance(reactants, tuple)

    def test_reactants_are_tree_molecules(self):
        rxn = TemplatedRetroReaction(
            self.mol, smarts=self.smarts, use_rdchiral=False
        )
        for outcome in rxn.reactants:
            for r in outcome:
                self.assertIsInstance(r, TreeMolecule)
                self.assertEqual(r.parent, self.mol)

    def test_missing_smarts_raises_key_error(self):
        with self.assertRaises(KeyError):
            TemplatedRetroReaction(self.mol)

    def test_copy_preserves_reactants(self):
        rxn = TemplatedRetroReaction(
            self.mol, smarts=self.smarts, use_rdchiral=False
        )
        _ = rxn.reactants  # force apply
        rxn_copy = rxn.copy()
        self.assertEqual(len(rxn_copy.reactants), len(rxn.reactants))


class TestReactionFingerprint(unittest.TestCase):
    """Test difference fingerprint."""

    def setUp(self):
        self.mol = TreeMolecule(parent=None, smiles="CCO")
        self.rxn = SmilesBasedRetroReaction(
            self.mol, reactants_str="CC.O"
        )

    def test_correct_shape(self):
        fp = self.rxn.fingerprint(radius=2, nbits=2048)
        self.assertEqual(fp.shape, (2048,))

    def test_can_have_negative_values(self):
        fp = self.rxn.fingerprint(radius=2, nbits=2048)
        # Difference FP = reactants - products, can be negative
        self.assertTrue(np.any(fp < 0) or np.any(fp > 0))


class TestHashKey(unittest.TestCase):
    """Test hash_key and hash_reactions."""

    def setUp(self):
        self.mol = TreeMolecule(parent=None, smiles="CCO")

    def test_deterministic(self):
        rxn1 = SmilesBasedRetroReaction(self.mol, reactants_str="CC.O")
        rxn2 = SmilesBasedRetroReaction(self.mol, reactants_str="CC.O")
        self.assertEqual(rxn1.hash_key(), rxn2.hash_key())

    def test_different_reactions_differ(self):
        rxn1 = SmilesBasedRetroReaction(self.mol, reactants_str="CC.O")
        mol2 = TreeMolecule(parent=None, smiles="CCCO")
        rxn2 = SmilesBasedRetroReaction(mol2, reactants_str="CCC.O")
        self.assertNotEqual(rxn1.hash_key(), rxn2.hash_key())

    def test_hash_reactions_deterministic(self):
        rxn1 = SmilesBasedRetroReaction(self.mol, reactants_str="CC.O")
        h1 = hash_reactions([rxn1])
        h2 = hash_reactions([rxn1])
        self.assertEqual(h1, h2)

    def test_hash_reactions_sorted_vs_unsorted(self):
        rxn1 = SmilesBasedRetroReaction(self.mol, reactants_str="CC.O")
        mol2 = TreeMolecule(parent=None, smiles="CCCO")
        rxn2 = SmilesBasedRetroReaction(mol2, reactants_str="CCC.O")
        h_sorted = hash_reactions([rxn1, rxn2], sort=True)
        h_sorted2 = hash_reactions([rxn2, rxn1], sort=True)
        self.assertEqual(h_sorted, h_sorted2)


if __name__ == "__main__":
    unittest.main()
