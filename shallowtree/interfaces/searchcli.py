from __future__ import annotations

import sys
from shallowtree.interfaces.full_tree_search import Expander
import argparse


def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Retrosynthetic analysis and scoring\n"
                    "Loads a list of SMILES from stdin, outputs a csv file with scores, predicted routes and required available starting materials",
        epilog="Example usage:\n"
               "echo 'Clc1ccccc1COC5CC(Nc3n[nH]c4cc(c2ccccc2)ccc34)C5' | searchcli --config config.yml --scaffold '[*]c1n[nH]c2cc(-c3ccccc3)ccc12' --depth 2 --routes\n"
               "searchcli --config config.yml --depth 2 --routes <smiles.txt >routes.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='path to configuration yml file'
    )

    parser.add_argument(
        '--depth',
        type=int,
        required=True,
        help='depth of retrosynthetic tree'
    )

    # Optional arguments
    parser.add_argument(
        '--scaffold',
        type=str,
        default=None,
        help='Scaffold SMILES for context search, e.g. \'[*]c1n[nH]c2cc(-c3ccccc3)ccc12\''
    )

    parser.add_argument(
        '--routes',
        action='store_true',
        help='Enable generation of synthetic routes'
    )

    if '-h' in sys.argv or '--help' in sys.argv:
        parser.print_help()
        sys.exit(0)

    # Parse arguments
    if sys.stdin.isatty():
        print('No SMILES input was detected!', file=sys.stderr)
        print('Use -h or --help for help', file=sys.stderr)
        sys.exit(1)
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

    df.to_csv(sys.stdout, index=False) # NOT for very large files that can overflow the stdout buffer, can use chunks etc
    sys.stdout.flush()


if __name__ == '__main__':
    main()
