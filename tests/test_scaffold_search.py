"""Tests for ScaffoldSearch — context-scaffold boundary handling and best_route."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest
from collections import defaultdict
from unittest.mock import MagicMock

from shallowtree.chem.molecules.tree_molecule import TreeMolecule
from shallowtree.interfaces.search_modes.scaffold_search import ScaffoldSearch
from tests.search_helpers import _make_search


class TestScaffoldBestRoute(unittest.TestCase):
    """Test ScaffoldSearch.best_route context-scaffold terminal handling."""

    def test_best_route_silent_for_context_scaffold_mol(self):
        """In scaffold search, scaffold-matching reactants are intentional
        terminal nodes (never added to solved); best_route must not warn."""
        from rdkit import Chem
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=False)
        exp = _make_search(ScaffoldSearch, stock=stock)
        exp._input_config.depth = 2
        exp.solved = {}
        context_scaffold = Chem.MolFromSmarts("CCO")

        mol = TreeMolecule(parent=None, smiles="CCO")
        tree = defaultdict(list)
        building_blocks = []
        with self.assertNoLogs(exp._logger, level="WARNING"):
            exp.best_route(mol, 0, tree, building_blocks, context_scaffold=context_scaffold)
        self.assertIn("CCO", building_blocks)

    def test_best_route_silent_for_relaxed_context_scaffold_mol(self):
        """The relaxed boundary check makes the scaffold-side reactant of a
        Williamson-style cut a terminal (the wildcard's atom is gone, leaving
        H). _matches_context_scaffold must recognise that via the stripped
        scaffold so best_route does not warn on it."""
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=False)
        exp = _make_search(ScaffoldSearch, stock=stock)
        exp._input_config.depth = 2
        exp.solved = {}

        scaffold = ScaffoldSearch._parse_scaffold_query("[*]Oc1ccccc1")
        info = ScaffoldSearch._scaffold_wildcard_info(scaffold)
        stripped = info[1]

        mol = TreeMolecule(parent=None, smiles="Oc1ccccc1")  # phenol
        tree = defaultdict(list)
        building_blocks = []
        with self.assertNoLogs(exp._logger, level="WARNING"):
            exp.best_route(mol, 0, tree, building_blocks, scaffold, stripped)
        self.assertIn("Oc1ccccc1", building_blocks)

    def test_scaffold_wildcard_info_only_for_single_leaf_wildcard(self):
        """The relaxed boundary check applies only when the scaffold has
        exactly one wildcard atom and it's a leaf (degree 1)."""

        # Single leaf wildcard — the relaxation applies.
        q = ScaffoldSearch._parse_scaffold_query("[*]Oc1ccccc1")
        info = ScaffoldSearch._scaffold_wildcard_info(q)
        self.assertIsNotNone(info)
        idx, stripped = info
        self.assertEqual(q.GetAtomWithIdx(idx).GetAtomicNum(), 0)
        self.assertEqual(stripped.GetNumAtoms(), q.GetNumAtoms() - 1)

        # No wildcard — relaxation does not apply.
        q = ScaffoldSearch._parse_scaffold_query("Oc1ccccc1")
        self.assertIsNone(ScaffoldSearch._scaffold_wildcard_info(q))

        # Two wildcards — ambiguous, do not relax.
        q = ScaffoldSearch._parse_scaffold_query("[*]Oc1ccc([*])cc1")
        self.assertIsNone(ScaffoldSearch._scaffold_wildcard_info(q))

        # Internal wildcard (degree > 1) — cutting it would split the
        # scaffold; the simple "scaffold-minus-wildcard" model breaks down.
        q = ScaffoldSearch._parse_scaffold_query("C[*]C")
        self.assertIsNone(ScaffoldSearch._scaffold_wildcard_info(q))

    def test_relaxed_boundary_accepts_williamson_retro(self):
        """A Williamson-style retro produces ArOH + alkyl-X. The phenol
        reactant matches scaffold-minus-wildcard, and the wildcard's atom
        (alpha CH2) is absent from the phenol — so the relaxed check fires."""
        from shallowtree.chem.reactions.templated_retro_reaction import TemplatedRetroReaction

        smi = "c1ccc(OCCc2ccccc2)cc1"
        scaffold = "[*]Oc1ccccc1"
        williamson = ("([C:2]-[CH2;D2;+0:1]-[O;H0;D2;+0:3]-[c:4])"
                      ">>(Br-[CH2;D2;+0:1]-[C:2]).([OH;D1;+0:3]-[c:4])")

        parent = TreeMolecule(parent=None, smiles=smi)
        action = TemplatedRetroReaction(
            parent, smarts=williamson,
            metadata={'classification': 'O-substitution', 'policy_name': 'rules'},
        )

        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(ScaffoldSearch, scaffold=scaffold, stock=stock)
        exp.expansion_policy.get_actions = MagicMock(return_value=([], []))
        exp.rules_expansion.get_actions = MagicMock(return_value=[action])

        exp._input_config.depth = 1
        df = exp.search([smi])
        self.assertGreater(df.iloc[0]['score'], 0.9)
        self.assertIn(parent.inchi_key, exp.solved)

    def test_relaxed_boundary_rejects_disconnection_elsewhere(self):
        """A retro that cuts a bond OUTSIDE the scaffold leaves the wildcard's
        atom alive in the scaffold-containing reactant. The relaxed check
        must reject this — otherwise the algorithm accepts any disconnection
        that happens to preserve the phenol core."""
        from shallowtree.chem.reactions.templated_retro_reaction import TemplatedRetroReaction

        smi = "c1ccc(OCCc2ccccc2)cc1"
        scaffold = "[*]Oc1ccccc1"
        # Cut the second-CH2 / second-phenyl bond (outside the scaffold);
        # produces Clc1ccccc1 + c1ccc(OCCCl)cc1. The latter still contains
        # the full scaffold (alpha CH2 intact), so it must NOT trigger.
        outside_template = ("([c:5]-[CH2;D2;+0:6])"
                            ">>(Cl-[c:5]).(Cl-[CH2;D2;+0:6])")

        parent = TreeMolecule(parent=None, smiles=smi)
        action = TemplatedRetroReaction(
            parent, smarts=outside_template,
            metadata={'classification': 'Aryl-alkyl cleavage', 'policy_name': 'rules'},
        )

        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(ScaffoldSearch, scaffold=scaffold, stock=stock)
        exp.expansion_policy.get_actions = MagicMock(return_value=([], []))
        exp.rules_expansion.get_actions = MagicMock(return_value=[action])

        exp._input_config.depth = 1
        df = exp.search([smi])
        self.assertEqual(df.iloc[0]['score'], 0.0)
        self.assertNotIn(parent.inchi_key, exp.solved)

    def test_relaxed_boundary_accepts_amide_coupling_amine_scaffold(self):
        """Amide coupling retro: R-C(=O)NH-R' -> R-COOH + H2N-R'. With the
        scaffold marking the amine side ([*]Nc1ccccc1), the wildcard sits
        where the carbonyl C attaches. The aniline reactant matches
        scaffold-minus-wildcard, and the wildcard's atom (carbonyl C) is
        absent from the aniline — so the relaxed check fires."""
        from shallowtree.chem.reactions.templated_retro_reaction import TemplatedRetroReaction

        smi = "CCC(=O)Nc1ccccc1"  # N-phenylpropanamide
        scaffold = "[*]Nc1ccccc1"
        amide = ("([C:1](=[O:2])-[NH;D2;+0:3]-[#6:4])"
                 ">>([OH]-[C:1]=[O:2]).([NH2;D1;+0:3]-[#6:4])")

        parent = TreeMolecule(parent=None, smiles=smi)
        action = TemplatedRetroReaction(
            parent, smarts=amide,
            metadata={'classification': 'Amide coupling', 'policy_name': 'rules'},
        )

        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(ScaffoldSearch, scaffold=scaffold, stock=stock)
        exp.expansion_policy.get_actions = MagicMock(return_value=([], []))
        exp.rules_expansion.get_actions = MagicMock(return_value=[action])

        exp._input_config.depth = 1
        df = exp.search([smi])
        self.assertGreater(df.iloc[0]['score'], 0.9)
        self.assertIn(parent.inchi_key, exp.solved)

    def test_relaxed_boundary_accepts_amide_coupling_acid_scaffold(self):
        """Same amide retro but with the scaffold marking the carboxylic
        acid side ([*]C(=O)c1ccccc1, wildcard at the amine N). The
        benzoic-acid reactant matches scaffold-minus-wildcard, and the
        wildcard's atom (the N) is absent from it — so the relaxed check
        fires from the acid side."""
        from shallowtree.chem.reactions.templated_retro_reaction import TemplatedRetroReaction

        smi = "CCNC(=O)c1ccccc1"  # N-ethylbenzamide
        scaffold = "[*]C(=O)c1ccccc1"
        amide = ("([C:1](=[O:2])-[NH;D2;+0:3]-[#6:4])"
                 ">>([OH]-[C:1]=[O:2]).([NH2;D1;+0:3]-[#6:4])")

        parent = TreeMolecule(parent=None, smiles=smi)
        action = TemplatedRetroReaction(
            parent, smarts=amide,
            metadata={'classification': 'Amide coupling', 'policy_name': 'rules'},
        )

        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_search(ScaffoldSearch, scaffold=scaffold, stock=stock)
        exp.expansion_policy.get_actions = MagicMock(return_value=([], []))
        exp.rules_expansion.get_actions = MagicMock(return_value=[action])

        exp._input_config.depth = 1
        df = exp.search([smi])
        self.assertGreater(df.iloc[0]['score'], 0.9)
        self.assertIn(parent.inchi_key, exp.solved)

    def test_parse_scaffold_query_handles_kekule_form(self):
        """Kekulé-written scaffolds must match aromatic targets the same way
        aromatic-lowercase scaffolds do, since MolFromSmarts takes bond orders
        literally while MolFromSmiles perceives aromaticity on the target."""
        from rdkit import Chem

        smi = ('CC1N=CSC=1C1C=C(OCC2C=CC(CN3CCN(C4C=C(C5C=CC=CC=5O)N=NC=4N)CC3)'
               '=CC=2)C(CNC(=O)[C@H]2N(C(=O)[C@@H](NC(=O)C3(F)CC3)C(C)(C)C)'
               'C[C@H](O)C2)=CC=1')
        kekule = ('CC1N=CSC=1C1C=C(O[*])C(CNC(=O)[C@H]2N(C(=O)[C@@H]'
                  '(NC(=O)C3(F)CC3)C(C)(C)C)C[C@H](O)C2)=CC=1')

        mol = Chem.MolFromSmiles(smi)
        q = ScaffoldSearch._parse_scaffold_query(kekule)
        self.assertGreater(len(mol.GetSubstructMatch(q)), 0)

        arom = '[*]c1n[nH]c2cc(-c3ccccc3)ccc12'
        smi2 = 'Clc1ccccc1COC5CC(Nc3n[nH]c4cc(c2ccccc2)ccc34)C5'
        mol2 = Chem.MolFromSmiles(smi2)
        q2 = ScaffoldSearch._parse_scaffold_query(arom)
        self.assertGreater(len(mol2.GetSubstructMatch(q2)), 0)


if __name__ == "__main__":
    unittest.main()
