"""
Tests for shallowtree.utils.image — molecule/route image generation and the
flat_route_to_dict adapter.

These tests need RDKit and Pillow but no TF models and no stock file: the
expected per-leaf in-stock truth is baked into tests/fixtures/route_fixtures.json
(generated from the benchmark128 ZINC run).
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import unittest

from shallowtree.chem.molecules.molecule import Molecule
from shallowtree.utils.image import (
    flat_route_to_dict,
    molecule_to_image,
    molecules_to_images,
    RouteImageFactory,
)

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "route_fixtures.json")


def _load_fixtures():
    with open(FIXTURE_PATH) as fileobj:
        data = json.load(fileobj)
    for fx in data["fixtures"]:
        # JSON keys are strings; the real search uses int depth keys
        fx["route"] = {int(k): v for k, v in fx["route"].items()}
    return data["fixtures"]


FIXTURES = _load_fixtures()
SOLVED = [f for f in FIXTURES if f["status"] == "solved"]
UNSOLVED = [f for f in FIXTURES if f["status"] != "solved"]


def _is_pil_image(obj):
    return hasattr(obj, "size") and hasattr(obj, "crop") and obj.size[0] > 0 and obj.size[1] > 0


class TestMoleculeDrawing(unittest.TestCase):
    """Tier 1: the low-level molecule drawing primitives."""

    def test_molecule_to_image_returns_image(self):
        img = molecule_to_image(Molecule(smiles="CCO"), "green")
        self.assertTrue(_is_pil_image(img))

    def test_molecules_to_images_one_per_molecule(self):
        mols = [Molecule(smiles="CCO"), Molecule(smiles="c1ccccc1")]
        imgs = molecules_to_images(mols, ["green", "orange"])
        self.assertEqual(len(imgs), len(mols))
        self.assertTrue(all(_is_pil_image(i) for i in imgs))


class TestFlatRouteToDict(unittest.TestCase):
    """The adapter that turns the search's depth-keyed route into a nested dict."""

    def test_root_inferred_from_shallowest_reaction(self):
        for fx in SOLVED:
            with self.subTest(idx=fx["idx"]):
                expected_root = fx["route"][1][0][0].split(" => ")[0].strip()
                nested = flat_route_to_dict(fx["route"])
                self.assertEqual(nested["smiles"], expected_root)
                self.assertEqual(nested["type"], "mol")

    def test_structure_alternates_mol_reaction(self):
        nested = flat_route_to_dict(SOLVED[0]["route"])
        self.assertEqual(nested["type"], "mol")
        self.assertIn("children", nested)
        reaction = nested["children"][0]
        self.assertEqual(reaction["type"], "reaction")
        self.assertTrue(all(c["type"] == "mol" for c in reaction["children"]))

    def test_in_stock_set_predicate(self):
        fx = SOLVED[0]
        in_stock = {smi for smi, ok in fx["leaf_in_stock"].items() if ok}
        nested = flat_route_to_dict(fx["route"], in_stock=in_stock)
        self._assert_leaf_coloring(nested, fx["leaf_in_stock"])

    def test_in_stock_callable_predicate(self):
        fx = SOLVED[0]
        truth = fx["leaf_in_stock"]
        nested = flat_route_to_dict(fx["route"], in_stock=lambda smi: truth.get(smi, False))
        self._assert_leaf_coloring(nested, truth)

    def test_in_stock_none_marks_all_leaves_in_stock(self):
        nested = flat_route_to_dict(SOLVED[0]["route"], in_stock=None)
        leaves = self._collect_leaves(nested)
        self.assertTrue(leaves)  # sanity: there are leaves
        self.assertTrue(all(leaf["in_stock"] for leaf in leaves))

    def test_internal_mol_nodes_not_in_stock(self):
        # A molecule that gets expanded (has a reaction child) is never a stock leaf
        nested = flat_route_to_dict(SOLVED[0]["route"], in_stock=lambda smi: True)
        self._assert_internal_not_in_stock(nested)

    def test_reproduces_stored_nested(self):
        # Adapter output must match the nested dict captured at generation time
        for fx in SOLVED:
            with self.subTest(idx=fx["idx"]):
                in_stock = {smi for smi, ok in fx["leaf_in_stock"].items() if ok}
                self.assertEqual(flat_route_to_dict(fx["route"], in_stock=in_stock), fx["nested"])

    def test_leaf_coloring_matches_ground_truth(self):
        for fx in SOLVED:
            with self.subTest(idx=fx["idx"]):
                in_stock = {smi for smi, ok in fx["leaf_in_stock"].items() if ok}
                nested = flat_route_to_dict(fx["route"], in_stock=in_stock)
                self._assert_leaf_coloring(nested, fx["leaf_in_stock"])

    def test_empty_route_raises(self):
        # Unsolved targets have an empty route -> nothing to draw
        self.assertTrue(UNSOLVED)  # sanity: we have unsolved fixtures
        for fx in UNSOLVED:
            with self.subTest(idx=fx["idx"]):
                self.assertFalse(fx["route"])
                with self.assertRaises(ValueError):
                    flat_route_to_dict(fx["route"])

    # -- helpers ----------------------------------------------------------
    def _collect_leaves(self, node, out=None):
        out = [] if out is None else out
        children = node.get("children")
        if not children:
            out.append(node)
            return out
        for reaction in children:
            for mol in reaction["children"]:
                self._collect_leaves(mol, out)
        return out

    def _assert_leaf_coloring(self, nested, truth):
        for leaf in self._collect_leaves(nested):
            self.assertEqual(leaf["in_stock"], truth[leaf["smiles"]],
                             f"leaf {leaf['smiles']} coloring mismatch")

    def _assert_internal_not_in_stock(self, node):
        children = node.get("children")
        if not children:
            return
        self.assertFalse(node["in_stock"], f"internal node {node['smiles']} marked in_stock")
        for reaction in children:
            for mol in reaction["children"]:
                self._assert_internal_not_in_stock(mol)


class TestRouteImageFactory(unittest.TestCase):
    """End-to-end: real benchmark routes render to a non-empty image."""

    def test_renders_each_solved_route(self):
        for fx in SOLVED:
            with self.subTest(idx=fx["idx"]):
                in_stock = {smi for smi, ok in fx["leaf_in_stock"].items() if ok}
                nested = flat_route_to_dict(fx["route"], in_stock=in_stock)
                img = RouteImageFactory(nested).image
                self.assertTrue(_is_pil_image(img))

    def test_depth3_route_has_three_reaction_levels(self):
        # benchmark used max_depth=2 -> routes up to 3 reaction steps (keys 1..3)
        deep = [f for f in SOLVED if f["maxdepth"] == 3]
        self.assertTrue(deep)
        for fx in deep:
            with self.subTest(idx=fx["idx"]):
                self.assertEqual(max(fx["route"].keys()), 3)


if __name__ == "__main__":
    unittest.main()
