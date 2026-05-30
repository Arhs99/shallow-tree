"""Tests for shallowtree.interfaces.full_tree_search — Expander tree search."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch

from shallowtree.chem.tree_molecule import TreeMolecule


def _make_expander(**overrides):
    """Create an Expander with mocked dependencies."""
    with patch("shallowtree.interfaces.full_tree_search.Expander._setup_filter_policy") as mock_fp, \
         patch("shallowtree.interfaces.full_tree_search.Expander._setup_expansion_policy") as mock_ep, \
         patch("shallowtree.interfaces.full_tree_search.Expander._setup_stock") as mock_st, \
         patch("shallowtree.interfaces.full_tree_search.Expander._setup_redis_cache") as mock_rc, \
         patch("shallowtree.interfaces.full_tree_search.Expander._setup_rules_expansion") as mock_re:

        mock_stock = MagicMock()
        mock_stock.__contains__ = MagicMock(return_value=False)

        mock_expansion = MagicMock()
        mock_expansion.get_actions = MagicMock(return_value=([], []))

        mock_rules = MagicMock()
        mock_rules.get_actions = MagicMock(return_value=[])

        mock_filter = MagicMock()

        mock_fp.return_value = mock_filter
        mock_ep.return_value = mock_expansion
        mock_st.return_value = mock_stock
        mock_rc.return_value = None
        mock_re.return_value = mock_rules

        mock_config = MagicMock()
        mock_config.search.score_acceptance_threshold = 0.9
        from shallowtree.interfaces.full_tree_search import Expander
        exp = Expander(mock_config)

        # Apply overrides
        for k, v in overrides.items():
            setattr(exp, k, v)

        return exp


def _make_action(reactant_smiles_list, classification="test", policy_name="ml",
                 feasibility=1.0):
    """Create a mock action with real TreeMolecule reactants."""
    parent = TreeMolecule(parent=None, smiles="CCO")
    reactants = tuple(
        TreeMolecule(parent=parent, smiles=smi) for smi in reactant_smiles_list
    )
    action = MagicMock()
    action.reactants = (reactants,)
    action.metadata = {
        "classification": classification,
        "policy_name": policy_name,
        "feasibility": feasibility,
    }
    return action


class TestReqSearchTree(unittest.TestCase):
    """Test req_search_tree logic."""

    def test_stock_mol_returns_1(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)
        mol = TreeMolecule(parent=None, smiles="CCO")
        score = exp.req_search_tree(mol, depth=0)
        self.assertEqual(score, 1.0)

    def test_stock_mol_cached_at_depth_0(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)
        mol = TreeMolecule(parent=None, smiles="CCO")
        exp.req_search_tree(mol, depth=0)
        self.assertIn(mol.inchi_key, exp.cache)
        self.assertEqual(exp.cache[mol.inchi_key], (0, 1.0))

    def test_depth_exceeds_max_returns_0(self):
        exp = _make_expander()
        exp.max_depth = 2
        mol = TreeMolecule(parent=None, smiles="CCO")
        score = exp.req_search_tree(mol, depth=3)
        self.assertEqual(score, 0.0)

    def test_cache_hit_reuse_when_cdepth_le_depth(self):
        exp = _make_expander()
        mol = TreeMolecule(parent=None, smiles="CCO")
        exp.cache[mol.inchi_key] = (1, 0.8)
        score = exp.req_search_tree(mol, depth=2)
        self.assertEqual(score, 0.8)

    def test_cache_skip_when_cdepth_gt_depth(self):
        """When cached depth > current depth, cache should be skipped."""
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)
        mol = TreeMolecule(parent=None, smiles="CCO")
        exp.cache[mol.inchi_key] = (2, 0.5)
        score = exp.req_search_tree(mol, depth=0)
        self.assertEqual(score, 1.0)

    def test_mean_of_reactants_scoring(self):
        """Score = mean of reactant scores."""
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)
        exp.max_depth = 2

        mol = TreeMolecule(parent=None, smiles="c1ccccc1CO")
        action = _make_action(["c1ccccc1", "CO"], policy_name="rules")

        exp.expansion_policy.get_actions = MagicMock(return_value=([], []))
        exp.rules_expansion.get_actions = MagicMock(return_value=[action])

        score = exp.req_search_tree(mol, depth=0)
        self.assertEqual(score, 1.0)

    def test_solved_threshold(self):
        """Only scores > 0.9 are added to self.solved."""
        mol = TreeMolecule(parent=None, smiles="c1ccccc1CO")
        reactant1 = TreeMolecule(parent=mol, smiles="c1ccccc1")
        reactant2 = TreeMolecule(parent=mol, smiles="CO")

        stock_inchis = {reactant1.inchi_key, reactant2.inchi_key}
        stock = MagicMock()
        stock.__contains__ = MagicMock(side_effect=lambda m: m.inchi_key in stock_inchis)
        exp = _make_expander(stock=stock)
        exp.max_depth = 2

        action = MagicMock()
        action.reactants = ((reactant1, reactant2),)
        action.metadata = {"classification": "test", "policy_name": "rules", "feasibility": 1.0}

        exp.expansion_policy.get_actions = MagicMock(return_value=([], []))
        exp.rules_expansion.get_actions = MagicMock(return_value=[action])

        exp.req_search_tree(mol, depth=0)
        self.assertIn(mol.inchi_key, exp.solved)

    def test_filter_threshold_gates_actions(self):
        """ML actions with feasibility < 0.5 should be excluded."""
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=False)
        exp = _make_expander(stock=stock)
        exp.max_depth = 2

        mol = TreeMolecule(parent=None, smiles="c1ccccc1CO")
        action = _make_action(["c1ccccc1", "CO"], policy_name="ml_policy")

        exp.expansion_policy.get_actions = MagicMock(return_value=([action], [0.5]))
        exp.rules_expansion.get_actions = MagicMock(return_value=[])

        mock_filter = MagicMock()
        mock_filter.batch_feasibility = MagicMock(return_value=[("", 0.3)])
        exp.filter_policy.__getitem__ = MagicMock(return_value=mock_filter)
        exp.filter_policy.selection = {"test_filter": True}

        score = exp.req_search_tree(mol, depth=0)
        self.assertEqual(score, 0.0)

    def test_rules_actions_bypass_filter(self):
        """Actions with policy_name='rules' bypass filter check."""
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)
        exp.max_depth = 2

        mol = TreeMolecule(parent=None, smiles="c1ccccc1CO")
        action = _make_action(["c1ccccc1", "CO"], policy_name="rules")

        exp.expansion_policy.get_actions = MagicMock(return_value=([], []))
        exp.rules_expansion.get_actions = MagicMock(return_value=[action])

        score = exp.req_search_tree(mol, depth=0)
        self.assertGreater(score, 0.9)


class TestBestRoute(unittest.TestCase):
    """Test best_route reconstruction."""

    def test_single_step_reconstruction(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)
        exp.max_depth = 2

        mol = TreeMolecule(parent=None, smiles="c1ccccc1CO")
        reactant1 = TreeMolecule(parent=mol, smiles="c1ccccc1")
        reactant2 = TreeMolecule(parent=mol, smiles="CO")
        exp.solved[mol.inchi_key] = ((reactant1, reactant2), 1.0, "test")

        tree = defaultdict(list)
        exp.BBs = []
        exp.best_route(mol, 0, tree)

        self.assertIn(1, tree)
        self.assertEqual(len(exp.BBs), 2)

    def test_unsolved_mol_becomes_bb(self):
        exp = _make_expander()
        exp.max_depth = 2

        mol = TreeMolecule(parent=None, smiles="CCO")
        tree = defaultdict(list)
        exp.BBs = []
        exp.best_route(mol, 0, tree)

        self.assertIn("CCO", exp.BBs)
        self.assertEqual(len(tree), 0)

    def test_best_route_warns_when_solved_missing_for_non_stock_mol(self):
        """Guard against silent route truncation from LRU eviction or redis
        partial writes: a non-stock mol missing from self.solved should warn."""
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=False)
        exp = _make_expander(stock=stock)
        exp.max_depth = 2
        exp.solved = {}

        mol = TreeMolecule(parent=None, smiles="CCO")
        tree = defaultdict(list)
        exp.BBs = []
        with self.assertLogs(exp._logger, level="WARNING") as cm:
            exp.best_route(mol, 0, tree)
        self.assertTrue(any("route truncated" in m for m in cm.output))
        self.assertIn("CCO", exp.BBs)

    def test_best_route_silent_for_stock_mol(self):
        """The normal stock-termination path stays silent (no warning)."""
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)
        exp.max_depth = 2
        exp.solved = {}

        mol = TreeMolecule(parent=None, smiles="CCO")
        tree = defaultdict(list)
        exp.BBs = []
        with self.assertNoLogs(exp._logger, level="WARNING"):
            exp.best_route(mol, 0, tree)
        self.assertIn("CCO", exp.BBs)

    def test_best_route_silent_for_context_scaffold_mol(self):
        """In context_search, scaffold-matching reactants are intentional
        terminal nodes (never added to solved); best_route must not warn."""
        from rdkit import Chem
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=False)
        exp = _make_expander(stock=stock)
        exp.max_depth = 2
        exp.solved = {}
        context_scaffold = Chem.MolFromSmarts("CCO")

        mol = TreeMolecule(parent=None, smiles="CCO")
        tree = defaultdict(list)
        exp.BBs = []
        with self.assertNoLogs(exp._logger, level="WARNING"):
            exp.best_route(mol, 0, tree, context_scaffold=context_scaffold)
        self.assertIn("CCO", exp.BBs)

    def test_best_route_silent_for_relaxed_context_scaffold_mol(self):
        """The relaxed boundary check makes the scaffold-side reactant of a
        Williamson-style cut a terminal (the wildcard's atom is gone, leaving
        H). _matches_context_scaffold must recognise that via the stripped
        scaffold so best_route does not warn on it."""
        from rdkit import Chem
        from shallowtree.interfaces.full_tree_search import Expander

        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=False)
        exp = _make_expander(stock=stock)
        exp.max_depth = 2
        exp.solved = {}

        scaffold = Expander._parse_scaffold_query("[*]Oc1ccccc1")
        # exp._context_scaffold = scaffold
        info = Expander._scaffold_wildcard_info(scaffold)
        stripped = info[1]

        mol = TreeMolecule(parent=None, smiles="Oc1ccccc1")  # phenol
        tree = defaultdict(list)
        exp.BBs = []
        with self.assertNoLogs(exp._logger, level="WARNING"):
            exp.best_route(mol, 0, tree, scaffold, stripped)
        self.assertIn("Oc1ccccc1", exp.BBs)

    def test_scaffold_wildcard_info_only_for_single_leaf_wildcard(self):
        """The relaxed boundary check applies only when the scaffold has
        exactly one wildcard atom and it's a leaf (degree 1)."""
        from shallowtree.interfaces.full_tree_search import Expander

        # Single leaf wildcard — the relaxation applies.
        q = Expander._parse_scaffold_query("[*]Oc1ccccc1")
        info = Expander._scaffold_wildcard_info(q)
        self.assertIsNotNone(info)
        idx, stripped = info
        self.assertEqual(q.GetAtomWithIdx(idx).GetAtomicNum(), 0)
        self.assertEqual(stripped.GetNumAtoms(), q.GetNumAtoms() - 1)

        # No wildcard — relaxation does not apply.
        q = Expander._parse_scaffold_query("Oc1ccccc1")
        self.assertIsNone(Expander._scaffold_wildcard_info(q))

        # Two wildcards — ambiguous, do not relax.
        q = Expander._parse_scaffold_query("[*]Oc1ccc([*])cc1")
        self.assertIsNone(Expander._scaffold_wildcard_info(q))

        # Internal wildcard (degree > 1) — cutting it would split the
        # scaffold; the simple "scaffold-minus-wildcard" model breaks down.
        q = Expander._parse_scaffold_query("C[*]C")
        self.assertIsNone(Expander._scaffold_wildcard_info(q))

    def test_relaxed_boundary_accepts_williamson_retro(self):
        """A Williamson-style retro produces ArOH + alkyl-X. The phenol
        reactant matches scaffold-minus-wildcard, and the wildcard's atom
        (alpha CH2) is absent from the phenol — so the relaxed check fires."""
        from shallowtree.chem.reaction import TemplatedRetroReaction

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
        exp = _make_expander(stock=stock)
        exp.expansion_policy.get_actions = MagicMock(return_value=([], []))
        exp.rules_expansion.get_actions = MagicMock(return_value=[action])

        df = exp.context_search([smi], scaffold_str=scaffold, max_depth=1)
        self.assertGreater(df.iloc[0]['score'], 0.9)
        self.assertIn(parent.inchi_key, exp.solved)

    def test_relaxed_boundary_rejects_disconnection_elsewhere(self):
        """A retro that cuts a bond OUTSIDE the scaffold leaves the wildcard's
        atom alive in the scaffold-containing reactant. The relaxed check
        must reject this — otherwise the algorithm accepts any disconnection
        that happens to preserve the phenol core."""
        from shallowtree.chem.reaction import TemplatedRetroReaction

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
        exp = _make_expander(stock=stock)
        exp.expansion_policy.get_actions = MagicMock(return_value=([], []))
        exp.rules_expansion.get_actions = MagicMock(return_value=[action])

        df = exp.context_search([smi], scaffold_str=scaffold, max_depth=1)
        self.assertEqual(df.iloc[0]['score'], 0.0)
        self.assertNotIn(parent.inchi_key, exp.solved)

    def test_relaxed_boundary_accepts_amide_coupling_amine_scaffold(self):
        """Amide coupling retro: R-C(=O)NH-R' -> R-COOH + H2N-R'. With the
        scaffold marking the amine side ([*]Nc1ccccc1), the wildcard sits
        where the carbonyl C attaches. The aniline reactant matches
        scaffold-minus-wildcard, and the wildcard's atom (carbonyl C) is
        absent from the aniline — so the relaxed check fires."""
        from shallowtree.chem.reaction import TemplatedRetroReaction

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
        exp = _make_expander(stock=stock)
        exp.expansion_policy.get_actions = MagicMock(return_value=([], []))
        exp.rules_expansion.get_actions = MagicMock(return_value=[action])

        df = exp.context_search([smi], scaffold_str=scaffold, max_depth=1)
        self.assertGreater(df.iloc[0]['score'], 0.9)
        self.assertIn(parent.inchi_key, exp.solved)

    def test_relaxed_boundary_accepts_amide_coupling_acid_scaffold(self):
        """Same amide retro but with the scaffold marking the carboxylic
        acid side ([*]C(=O)c1ccccc1, wildcard at the amine N). The
        benzoic-acid reactant matches scaffold-minus-wildcard, and the
        wildcard's atom (the N) is absent from it — so the relaxed check
        fires from the acid side."""
        from shallowtree.chem.reaction import TemplatedRetroReaction

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
        exp = _make_expander(stock=stock)
        exp.expansion_policy.get_actions = MagicMock(return_value=([], []))
        exp.rules_expansion.get_actions = MagicMock(return_value=[action])

        df = exp.context_search([smi], scaffold_str=scaffold, max_depth=1)
        self.assertGreater(df.iloc[0]['score'], 0.9)
        self.assertIn(parent.inchi_key, exp.solved)

    def test_parse_scaffold_query_handles_kekule_form(self):
        """Kekulé-written scaffolds must match aromatic targets the same way
        aromatic-lowercase scaffolds do, since MolFromSmarts takes bond orders
        literally while MolFromSmiles perceives aromaticity on the target."""
        from rdkit import Chem
        from shallowtree.interfaces.full_tree_search import Expander

        smi = ('CC1N=CSC=1C1C=C(OCC2C=CC(CN3CCN(C4C=C(C5C=CC=CC=5O)N=NC=4N)CC3)'
               '=CC=2)C(CNC(=O)[C@H]2N(C(=O)[C@@H](NC(=O)C3(F)CC3)C(C)(C)C)'
               'C[C@H](O)C2)=CC=1')
        kekule = ('CC1N=CSC=1C1C=C(O[*])C(CNC(=O)[C@H]2N(C(=O)[C@@H]'
                  '(NC(=O)C3(F)CC3)C(C)(C)C)C[C@H](O)C2)=CC=1')

        mol = Chem.MolFromSmiles(smi)
        q = Expander._parse_scaffold_query(kekule)
        self.assertGreater(len(mol.GetSubstructMatch(q)), 0)

        arom = '[*]c1n[nH]c2cc(-c3ccccc3)ccc12'
        smi2 = 'Clc1ccccc1COC5CC(Nc3n[nH]c4cc(c2ccccc2)ccc34)C5'
        mol2 = Chem.MolFromSmiles(smi2)
        q2 = Expander._parse_scaffold_query(arom)
        self.assertGreater(len(mol2.GetSubstructMatch(q2)), 0)


class TestSearchTree(unittest.TestCase):
    """Test search_tree returns correct DataFrame."""

    def test_returns_dataframe_with_correct_columns(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)

        df = exp.search_tree(["CCO"], max_depth=2)
        self.assertIn("SMILES", df.columns)
        self.assertIn("score", df.columns)
        self.assertIn("route", df.columns)
        self.assertIn("BBs", df.columns)

    def test_handles_multiple_smiles(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)

        df = exp.search_tree(["CCO", "CCCO"], max_depth=2)
        self.assertEqual(len(df), 2)

    def test_works_with_no_redis_cache(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)
        exp.redis_cache = None

        df = exp.search_tree(["CCO"], max_depth=2)
        self.assertEqual(len(df), 1)


class TestCacheCorrectness(unittest.TestCase):
    """Test cache behavior."""

    def test_cache_populated_after_search(self):
        stock = MagicMock()
        stock.__contains__ = MagicMock(return_value=True)
        exp = _make_expander(stock=stock)

        mol = TreeMolecule(parent=None, smiles="CCO")
        exp.req_search_tree(mol, depth=0)
        self.assertIn(mol.inchi_key, exp.cache)

    def test_depth_stored_correctly(self):
        mol = TreeMolecule(parent=None, smiles="c1ccccc1CO")
        reactant1 = TreeMolecule(parent=mol, smiles="c1ccccc1")
        reactant2 = TreeMolecule(parent=mol, smiles="CO")

        stock_inchis = {reactant1.inchi_key, reactant2.inchi_key}
        stock = MagicMock()
        stock.__contains__ = MagicMock(side_effect=lambda m: m.inchi_key in stock_inchis)
        exp = _make_expander(stock=stock)
        exp.max_depth = 2

        action = MagicMock()
        action.reactants = ((reactant1, reactant2),)
        action.metadata = {"classification": "test", "policy_name": "rules", "feasibility": 1.0}
        exp.expansion_policy.get_actions = MagicMock(return_value=([], []))
        exp.rules_expansion.get_actions = MagicMock(return_value=[action])

        exp.req_search_tree(mol, depth=1)
        depth, _ = exp.cache[mol.inchi_key]
        self.assertEqual(depth, 1)


if __name__ == "__main__":
    unittest.main()
