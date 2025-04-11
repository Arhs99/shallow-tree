""" Module containing classes and routines for the CLI
"""

from __future__ import annotations

import sys
from shallowtree.interfaces.full_tree_search import Expander
import argparse


def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Retrosynthetic analysis and scoring"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to config.yml"
    )

    parser.add_argument(
        "--depth",
        type=int,
        required=True,
        help="Search depth for molecular analysis"
    )

    # Optional arguments
    parser.add_argument(
        "--scaffold",
        type=str,
        default=None,
        help="Scaffold SMILES string to use as reference (optional)"
    )

    parser.add_argument(
        "--routes",
        action="store_true",
        help="Enable generation of synthetic routes"
    )

    # Parse arguments
    smiles = [x.strip() for x in sys.stdin]
    args = parser.parse_args()
    expander = Expander(configfile=args.config)
    expander.expansion_policy.select_first()
    expander.filter_policy.select_first()
    expander.stock.select_first()

    if args.scaffold is None:
        df = expander.search_tree(smiles, max_depth=args.depth)
    else:
        df = expander.context_search(smiles, args.scaffold, max_depth=args.depth)

    if not args.routes:
        df = df.drop(columns=['route'])

    print(df.to_csv())
if __name__ == "__main__":
    main()