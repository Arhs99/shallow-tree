#!/usr/bin/env python
"""
Smoke test for the inherited image-generation code (shallowtree/utils/image.py).

shallow-tree copied aizynthfinder's drawing code but refactored away the
ReactionTree.to_dict producer that fed it. This script checks which tiers still
work, mirroring the original to_dict -> RouteImageFactory.to_image contract.

Run:
    conda run -n shallow-tree python scripts/check_image.py
    # optional end-to-end check against a real search (loads models):
    conda run -n shallow-tree python scripts/check_image.py \
        --config application_config/config.json --smiles CCOC(=O)c1ccccc1

Exits non-zero if any check fails. Saves PNGs to a temp dir it prints.
"""
import argparse
import sys
import tempfile
import os

from shallowtree.chem.mol import Molecule
from shallowtree.utils.image import (
    molecule_to_image,
    molecules_to_images,
    flat_route_to_dict,
    RouteImageFactory,
)

OUT_DIR = tempfile.mkdtemp(prefix="shallowtree_image_check_")


def _save(img, name):
    path = os.path.join(OUT_DIR, name)
    img.save(path)
    return path


def check_molecule_drawing():
    """Tier 1: single + grid molecule images (PIL + rdkit Draw only)."""
    img = molecule_to_image(Molecule(smiles="CCO"), "green")
    assert img.size[0] > 0 and img.size[1] > 0, "molecule_to_image produced empty image"
    _save(img, "mol_single.png")

    imgs = molecules_to_images(
        [Molecule(smiles="CCO"), Molecule(smiles="c1ccccc1")],
        ["green", "orange"],
    )
    assert len(imgs) == 2, "molecules_to_images returned wrong count"
    assert all(i.size[0] > 0 for i in imgs), "molecules_to_images produced empty image"
    for idx, i in enumerate(imgs):
        _save(i, f"mol_grid_{idx}.png")
    print("  [OK] molecule drawing (molecule_to_image, molecules_to_images)")


def check_route_render_fixture():
    """Tier 2a: RouteImageFactory against a hand-built to_dict-style fixture."""
    # Ethyl benzoate -> ethanol + benzoic acid (1 step, 2 reactants)
    fixture = {
        "type": "mol",
        "hide": False,
        "smiles": "CCOC(=O)c1ccccc1",
        "is_chemical": True,
        "in_stock": False,
        "children": [
            {
                "type": "reaction",
                "hide": False,
                "smiles": "",
                "is_reaction": True,
                "metadata": {"classification": "esterification"},
                "children": [
                    {"type": "mol", "hide": False, "smiles": "CCO",
                     "is_chemical": True, "in_stock": True},
                    {"type": "mol", "hide": False, "smiles": "OC(=O)c1ccccc1",
                     "is_chemical": True, "in_stock": True},
                ],
            }
        ],
    }
    img = RouteImageFactory(fixture).image
    assert img.size[0] > 0 and img.size[1] > 0, "RouteImageFactory produced empty image"
    _save(img, "route_fixture.png")
    print("  [OK] route rendering from synthetic to_dict fixture")


def check_adapter():
    """Tier 2b: flat search route -> nested dict -> RouteImageFactory."""
    # Shape of the search's `route` column: {depth: [["prod => r1.r2", clas], ...]}
    flat_route = {
        1: [["CCOC(=O)c1ccccc1 => CCO.OC(=O)c1ccccc1", "esterification"]],
    }
    nested = flat_route_to_dict(flat_route, root_smiles="CCOC(=O)c1ccccc1")
    assert nested["type"] == "mol" and nested["smiles"] == "CCOC(=O)c1ccccc1"
    assert nested["children"][0]["type"] == "reaction"
    leaves = nested["children"][0]["children"]
    assert {leaf["smiles"] for leaf in leaves} == {"CCO", "OC(=O)c1ccccc1"}
    assert all(leaf["in_stock"] for leaf in leaves), "leaves should default to in_stock"

    img = RouteImageFactory(nested).image
    assert img.size[0] > 0 and img.size[1] > 0
    _save(img, "route_from_flat.png")
    print("  [OK] flat_route_to_dict adapter + render")


def check_real_search(config_path, smiles):
    """Tier 3 (opt-in): real search -> flat route -> adapter -> render."""
    from shallowtree.configs.application_configuration import ApplicationConfiguration
    from shallowtree.context.config import Configuration
    from shallowtree.interfaces.full_tree_search import Expander

    config_dict = Configuration.from_json(config_path)
    expander = Expander(ApplicationConfiguration(**config_dict))
    expander.expansion_policy.select_first()
    expander.filter_policy.select_first()
    expander.stock.select_first()

    df = expander.search_tree([smiles], max_depth=2)
    row = df.iloc[0]
    route = row["route"]
    if not route:
        print(f"  [SKIP] real search returned no route for {smiles} "
              f"(score={row['score']}) — nothing to render")
        return

    def _in_stock(smi):
        return Molecule(smiles=smi) in expander.stock

    # root_smiles is inferred from the route (canonical), not row["SMILES"] (raw input)
    nested = flat_route_to_dict(route, in_stock=_in_stock)
    assert nested.get("children"), "expected a multi-node route, got a bare leaf"
    img = RouteImageFactory(nested).image
    assert img.size[0] > 0 and img.size[1] > 0
    _save(img, "route_real_search.png")
    print(f"  [OK] real search route rendered for {smiles}")


def main():
    parser = argparse.ArgumentParser(description="Smoke test image generation")
    parser.add_argument("--config", help="path to config json for the opt-in real-search check")
    parser.add_argument("--smiles", default="CCOC(=O)c1ccccc1",
                        help="target SMILES for the real-search check")
    args = parser.parse_args()

    print(f"Writing images to: {OUT_DIR}")
    checks = [
        ("molecule drawing", check_molecule_drawing),
        ("route fixture render", check_route_render_fixture),
        ("flat-route adapter", check_adapter),
    ]
    if args.config:
        checks.append(("real search render", lambda: check_real_search(args.config, args.smiles)))

    failed = False
    for label, fn in checks:
        try:
            fn()
        except Exception as exc:  # noqa - report and continue to next check
            failed = True
            print(f"  [FAIL] {label}: {type(exc).__name__}: {exc}")
    if failed:
        print("\nImage generation is PARTIALLY broken — see failures above.")
        sys.exit(1)
    print(f"\nAll image checks passed. Inspect the PNGs in {OUT_DIR}")


if __name__ == "__main__":
    main()
