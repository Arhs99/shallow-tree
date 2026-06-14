"""Build a building-block stock (a set of InChI keys) from SMILES files.

This is a parallel, gzip-aware tool for turning a (possibly large, multi-column)
list of SMILES into the HDF5 stock format consumed by
``shallowtree.context.stock.queries.InMemoryInchiKeyQuery`` -- a pandas DataFrame
with a single ``inchi_key`` column stored under the key ``"table"`` (the same
format produced by ``shallowtree.tools.make_stock.make_hdf5_stock``).

Compared to ``smiles2stock`` it adds: multiprocessing, transparent ``.gz`` input,
column/header handling for tabular SMILES files, optional largest-fragment
desalting, and an optional ``inchi_key -> id`` provenance map.

The InChI key is computed exactly the way the search computes a molecule's key at
lookup time (``Chem.MolToInchiKey`` on the sanitized molecule -- see
``shallowtree.chem.molecules.molecule.Molecule.inchi_key``). No neutralization or
other standardization is applied beyond optional fragment selection, since any
such change would shift keys and silently break stock matching.

Examples
--------
Plain one-SMILES-per-line file::

    build_stock --files mols.smi --output stock.hdf5

Gzipped, tab/space-delimited file with a header and an id column, also writing a
provenance map::

    build_stock --files catalog.smi.gz --skip-header \\
        --smiles-col 0 --id-col 1 \\
        --output stock.hdf5 --id-map provenance.parquet
"""
from __future__ import annotations

import argparse
import gzip
import os
from multiprocessing import Pool, cpu_count
from typing import IO, List, Optional, Tuple

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

# Per-worker config, populated by the pool initializer (avoids pickling per call).
_CFG: dict = {}


def _open(path: str) -> IO:
    """Open a text file, transparently handling gzip by extension."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def _init_worker(smiles_col: int, id_col: Optional[int], delimiter: Optional[str],
                 desalt: bool) -> None:
    _CFG.update(
        smiles_col=smiles_col, id_col=id_col, delimiter=delimiter, desalt=desalt
    )


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Return the largest fragment by heavy-atom count (tie-break: exact MW)."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) == 1:
        return frags[0]
    return max(frags, key=lambda f: (f.GetNumHeavyAtoms(), Descriptors.ExactMolWt(f)))


def _process_line(line: str) -> Optional[Tuple[str, Optional[str]]]:
    """Parse one line -> (inchi_key, id) or None on empty/unparseable input."""
    parts = line.split(_CFG["delimiter"])
    if len(parts) <= _CFG["smiles_col"]:
        return None
    smiles = parts[_CFG["smiles_col"]]
    mol = Chem.MolFromSmiles(smiles)  # sanitizes by default
    if mol is None:
        return None
    if _CFG["desalt"]:
        mol = _largest_fragment(mol)
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:  # pylint: disable=broad-except
        return None
    if not key:
        return None
    id_col = _CFG["id_col"]
    rid = parts[id_col] if id_col is not None and len(parts) > id_col else None
    return key, rid


def build_stock(
    files: List[str],
    output: str,
    smiles_col: int = 0,
    id_col: Optional[int] = None,
    delimiter: Optional[str] = None,
    skip_header: bool = False,
    desalt: bool = True,
    id_map: Optional[str] = None,
    workers: Optional[int] = None,
) -> None:
    """Convert SMILES file(s) into a deduplicated HDF5 InChI-key stock.

    :param files: input file paths (``.gz`` handled transparently)
    :param output: path of the HDF5 stock to write
    :param smiles_col: zero-based column index of the SMILES
    :param id_col: optional column index of an identifier (for the provenance map)
    :param delimiter: column delimiter; ``None`` splits on any whitespace
    :param skip_header: skip the first line of each file
    :param desalt: keep only the largest fragment of multi-fragment entries
    :param id_map: optional path for an ``inchi_key -> id`` map (needs ``id_col``);
        ``.parquet`` is used when possible, otherwise a ``.csv.gz`` fallback
    :param workers: number of worker processes (defaults to ``cpu_count() - 2``)
    """
    if id_map is not None and id_col is None:
        raise ValueError("--id-map requires --id-col")
    workers = workers or max(1, cpu_count() - 2)

    keys: List[str] = []
    ids: List[Optional[str]] = []
    n = n_fail = 0

    with Pool(
        workers, initializer=_init_worker,
        initargs=(smiles_col, id_col, delimiter, desalt),
    ) as pool:
        for path in files:
            print(f"Processing {path}", flush=True)
            with _open(path) as fileobj:
                if skip_header:
                    next(fileobj, None)
                for res in pool.imap_unordered(_process_line, fileobj, chunksize=2000):
                    n += 1
                    if res is None:
                        n_fail += 1
                    else:
                        keys.append(res[0])
                        ids.append(res[1])
                    if n % 500000 == 0:
                        print(f"  read {n:,} | kept {len(keys):,} | failed {n_fail:,}",
                              flush=True)

    print(f"read {n:,} lines | parse/key failures {n_fail:,}", flush=True)

    df = pd.DataFrame({"inchi_key": keys, "id": ids})
    stock = df.drop_duplicates("inchi_key")[["inchi_key"]]
    stock.to_hdf(output, "table")
    print(f"Created HDF5 stock with {len(stock):,} unique compounds -> {output}",
          flush=True)

    if id_map is not None:
        prov = df.dropna(subset=["id"]).drop_duplicates()
        _write_id_map(prov, id_map)


def _write_id_map(prov: pd.DataFrame, path: str) -> None:
    if path.endswith(".parquet"):
        try:
            prov.to_parquet(path)
            print(f"Wrote {len(prov):,} inchi_key->id rows -> {path}", flush=True)
            return
        except Exception as exc:  # pylint: disable=broad-except
            path = os.path.splitext(path)[0] + ".csv.gz"
            print(f"parquet unavailable ({exc}); writing CSV instead", flush=True)
    compression = "gzip" if path.endswith(".gz") else "infer"
    prov.to_csv(path, index=False, compression=compression)
    print(f"Wrote {len(prov):,} inchi_key->id rows -> {path}", flush=True)


def _get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("build_stock")
    parser.add_argument("--files", required=True, nargs="+",
                        help="input file(s) with SMILES; .gz is handled transparently")
    parser.add_argument("--output", required=True,
                        help="path of the HDF5 stock to write")
    parser.add_argument("--smiles-col", type=int, default=0,
                        help="zero-based column index of the SMILES (default: 0)")
    parser.add_argument("--id-col", type=int, default=None,
                        help="optional column index of an id, for the provenance map")
    parser.add_argument("--delimiter", default=None,
                        help="column delimiter (default: any whitespace)")
    parser.add_argument("--skip-header", action="store_true",
                        help="skip the first line of each input file")
    parser.add_argument("--desalt", choices=["largest", "none"], default="largest",
                        help="multi-fragment handling: keep largest fragment, or none "
                             "(default: largest)")
    parser.add_argument("--id-map", default=None,
                        help="optional path for an inchi_key->id map (requires --id-col)")
    parser.add_argument("--workers", type=int, default=None,
                        help="number of worker processes (default: cpu_count() - 2)")
    return parser.parse_args()


def main() -> None:
    """Entry-point for the build_stock tool."""
    args = _get_arguments()
    build_stock(
        files=args.files,
        output=args.output,
        smiles_col=args.smiles_col,
        id_col=args.id_col,
        delimiter=args.delimiter,
        skip_header=args.skip_header,
        desalt=args.desalt == "largest",
        id_map=args.id_map,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
