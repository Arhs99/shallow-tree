import copy
from typing import List

import pandas as pd
from pathos.multiprocessing import ProcessPool
from rdkit import Chem

from shallowtree.configs.application_configuration import ApplicationConfiguration
from shallowtree.configs.input_configuration import InputConfiguration
from shallowtree.configs.stock_configuration import StockConfiguration
from shallowtree.context.config import Configuration
from shallowtree.context.stock.queries import InMemoryInchiKeyQuery
from shallowtree.context.stock.shared_inchi_key_set import SharedInchiKeySet
from shallowtree.context.stock.shared_stock_query import SharedInchiKeyQuery
from shallowtree.context.stock.stock import Stock
from shallowtree.interfaces.search_modes.tree_search import TreeSearch


def clone_config(input_config: InputConfiguration, smiles:List[str]):
    clone = copy.deepcopy(input_config)
    clone.smiles = smiles
    return clone


def _build_shared_stock(stock_configs: List[StockConfiguration]) -> SharedInchiKeySet:
    """Load all stock inchi keys in the parent and pack them into a single
    SharedInchiKeySet. Workers attach to this block by name, paying for one
    physical copy in /dev/shm rather than one heap copy per worker."""
    keys: List[str] = []
    for cfg in stock_configs:
        q = InMemoryInchiKeyQuery(
            path=cfg.dataset,
            inchi_key_col=cfg.inchi_key_col,
            price_col=cfg.price_col,
        )
        keys.extend(q.stock_inchikeys)
    shared = SharedInchiKeySet.build(keys)
    return shared

def _build_worker_stock(shm_name: str, shm_count: int) -> Stock:
    """Worker-side: attach to the shared inchi-key block and wrap it in a Stock."""
    attached = SharedInchiKeySet.attach(shm_name, shm_count)
    stock = Stock([])
    stock.load(SharedInchiKeyQuery(attached), key="shared")
    return stock


def parallel_search(input_config: InputConfiguration):
    config_dict = Configuration.from_json(input_config.app_configuration_path)
    app_config = ApplicationConfiguration(**config_dict)
    smiles = [input_config.smiles[i:i + input_config.parallel_processes]
              for i in range(0, len(input_config.smiles), input_config.parallel_processes)]
    input_configs = [clone_config(input_config, s) for s in smiles]

    shared = _build_shared_stock(app_config.stock)
    shm_name, shm_count = shared.shm_name, len(shared)

    def parallel_run(input_c: InputConfiguration):
        stock = _build_worker_stock(shm_name, shm_count)
        input_c.prebuilt_stock = stock

        search = TreeSearch(input_c)
        df = search.search(input_c.smiles)

        if not input_c.routes:
            df = df.drop(columns=['route'])
        return df

    try:
        pool = ProcessPool(nodes=input_config.parallel_processes)
        results = pool.map(parallel_run, input_configs)
        concatenated = pd.concat(results)
        return concatenated
    finally:
        shared.unlink()


def sequential_search(input_config: InputConfiguration):
    config_dict = Configuration.from_json(input_config.app_configuration_path)
    app_config = ApplicationConfiguration(**config_dict)

    shared = _build_shared_stock(app_config.stock)
    shm_name, shm_count = shared.shm_name, len(shared)
    stock = _build_worker_stock(shm_name, shm_count)
    input_config.prebuilt_stock = stock

    search = TreeSearch(input_config)
    df = search.search(input_config.smiles)

    if not input_config.routes:
        df = df.drop(columns=['route'])
    return df


def iterative_deepening_search(input_config: InputConfiguration):
    # Sequential iterative-deepening: build the stock once, then per target sweep
    # max_depth from d_start to d_max (defaults to ``depth``) on one warm search
    # instance, reporting the minimal resolving depth. Mirrors sequential_search.
    config_dict = Configuration.from_json(input_config.app_configuration_path)
    app_config = ApplicationConfiguration(**config_dict)

    shared = _build_shared_stock(app_config.stock)
    shm_name, shm_count = shared.shm_name, len(shared)
    stock = _build_worker_stock(shm_name, shm_count)
    input_config.prebuilt_stock = stock

    search = TreeSearch(input_config)
    df = search.search_iterative(input_config.smiles)

    if not input_config.routes:
        df = df.drop(columns=['route'])
    return df


# Per-worker warm state for parallel iterative deepening. Populated lazily INSIDE
# each worker process on its first task and reused across the targets that worker
# pulls, so the ONNX models load once per worker and self.cache/self.solved stay
# warm across targets. Keyed by shm_name: pathos reuses worker processes across
# pool invocations, so a new run (new shared-stock block => new shm_name) must
# rebuild the instance against the new stock — keying on shm_name makes that
# self-correcting. The parent NEVER touches this dict (fork hygiene: no TreeSearch
# in the parent before fork).
_WORKER: dict = {}


def _heavy_atom_count(smiles: str) -> int:
    """Cheap per-target cost proxy for longest-first scheduling. Unparseable
    SMILES sort last (count 0); input.py canonicalizes upstream so this is rare."""
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def _run_one_target(worker_config):
    # Build (or rebuild on a new run) the warm search instance strictly here, in
    # the worker, never in the parent.
    search = _WORKER["search"]
    try:
        row = search.search_iterative(worker_config.smiles).iloc[0].to_dict()
        row["error"] = None
    except Exception as exc:  # noqa: BLE001 - deliberately broad; isolate one target
        row = {"SMILES": worker_config.smiles[0], "score": 0.0, "resolved": False, "route": {},
               "BBs": [], "resolved_depth": None, "error": str(exc)}
    return row


def parallel_iterative_deepening_search(input_config: InputConfiguration):

    config_dict = Configuration.from_json(input_config.app_configuration_path)
    app_config = ApplicationConfiguration(**config_dict)

    shared = _build_shared_stock(app_config.stock)
    shm_name, shm_count = shared.shm_name, len(shared)
    stock = _build_worker_stock(shm_name, shm_count)
    input_config.prebuilt_stock = stock

    search = TreeSearch(input_config)
    _WORKER["search"] = search

    smiles = input_config.smiles
    # Longest-first dispatch (LPT): start the costliest sweeps at t=0 so the
    # easy-target tail overlaps them. ``order`` maps dispatch position -> original
    # index, used below to restore input order in the result frame.
    order = sorted(range(len(smiles)), key=lambda i: _heavy_atom_count(input_config.smiles[i]), reverse=True)

    input_configs = [clone_config(input_config, [smiles[i]]) for i in order]

    try:
        pool = ProcessPool(nodes=input_config.parallel_processes)
        results = pool.map(_run_one_target, input_configs)
    finally:
        shared.unlink()

    # results[k] is the row for original index order[k]; restore input order.
    rows = [None] * len(smiles)
    for k, i in enumerate(order):
        rows[i] = results[k]
    df = pd.DataFrame(rows).reset_index(drop=True)

    if not input_config.routes:
        df = df.drop(columns=['route'])
    return df