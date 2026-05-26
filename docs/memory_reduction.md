# Memory reduction

Summary of memory and runtime improvements landed on the `mem-reduction`
branch. Workload reference: 1000 high-MW SMILES from ChEMBL, depth=2,
ONNX expansion/filter models, ZINC stock (17.4M InChI keys).

## What changed

### 1. Packed stock representation

`InMemoryInchiKeyQuery` previously held the ZINC stock as a
`frozenset[str]` of 17.4M InChI keys. Replaced with `PackedInchiKeySet`:
a sorted numpy `|S27` array searched via `np.searchsorted`.

- Resident size: **1.99 GB → 567 MB** (−1.4 GB).
- `__contains__` is ~25× slower per call than a `frozenset` hash
  lookup (125 ns → 3.2 μs), but the cost is negligible against the
  RDKit work that surrounds every lookup.

Files: `shallowtree/context/stock/packed_inchi_key_set.py`,
`shallowtree/context/stock/queries.py`.

### 2. Shared stock across parallel workers

`multiprocessing.shared_memory`-backed `SharedInchiKeySet`. The parent
process builds the packed stock once; workers attach to it by name.
No per-worker reload, no per-worker resident duplication.

- Parent builds the shared block in ~16 s once, regardless of worker
  count.
- Workers carry the stock as shared (read-only) pages rather than a
  private 567 MB copy each.

Files: `shallowtree/context/stock/shared_inchi_key_set.py`,
`shallowtree/context/stock/shared_stock_query.py`,
`shallowtree/interfaces/parallel.py`,
`shallowtree/interfaces/full_tree_search.py` (`prebuilt_stock`
parameter on `Expander.__init__`).

### 3. TreeMolecule interning by InChI key

`_apply_with_rdkit` produces large numbers of duplicate reactant
TreeMolecules (common building blocks recur 1000+ times within a
single SMILES search). A per-`Expander` LRU cache keyed by InChI key
returns the previously-constructed instance instead of building a new
one.

- Measured dedup ratio: 5.97× on 1 SMILES, 5.21× on 10 SMILES.
- Cache bound: `LRUCache(2000)` — the multiplicity histogram showed
  ~83% of dedup is in the top few hundred most-duplicated keys, so a
  small bounded LRU captures nearly all of the win.
- Skips the expensive `mapped_mol` deep copy, `mapped_smiles` SMILES
  round-trip, and `remove_atom_mapping` for every cache hit.

Files: `shallowtree/chem/reaction.py` (`_make_or_intern_reactant`),
`shallowtree/chem/mol.py` (`intern_cache` plumbed through the
`TreeMolecule` parent chain), `shallowtree/utils/lru.py`.

## Benchmarks

### Single-process, 1000 SMILES (benchmark1K)

| Run | Peak RSS | Wall | Solved |
|---|---|---|---|
| Baseline (`main`, no LRU) | 16.03 GB | 1:32:38 | 56 |
| Packed stock | 16.34 GB | 1:32:46 | 56 |
| Packed stock + interning | ~17.1 GB | **1:04:00** | 56 |

`top_score` is bit-identical to the baseline for the packed-stock-only
run. Interning is **−31% wall clock**; the small RSS regression at
single-process scale is glibc arena fragmentation from `np.searchsorted`
allocation churn and does not appear at parallel scale (see below).

### 2-worker parallel, 1000 SMILES

| Run | Sum peak RSS | Wall | Solved |
|---|---|---|---|
| Unbounded baseline (per-worker stock) | 21.01 GB | 51:43 | 52 |
| Shared stock | 21.76 GB | **49:55** | 52 |

Bit-identical `top_score` vs the unbounded baseline. Wall clock −1:48
(~3%) from removing per-worker stock load.

### 8-worker parallel, 1000 SMILES (packed + shared + interning)

| Metric | 2-worker (shared only) | 8-worker (all changes) |
|---|---|---|
| Wall clock | 49:55 | **14:02** |
| Per-worker peak RSS | 10.7 GB | **5.04 GB** |
| Solved | 52 / 1000 | 51 / 1000 |

8-worker is **−71% wall clock** vs the 2-worker shared baseline. Per-
worker peak RSS drops by ~5.7 GB because the single-process arena
fragmentation seen with interning does not accumulate across only
125 SMILES per worker.

### Recommended worker counts (30 GB RAM)

- **4 workers** — comfortable headroom (~22 GB physical), no swap.
- **6 workers** — fits in RAM (≈ 31 GB physical), no swap.
- **8 workers** — best wall clock, engages swap (~10 GB paged out
  during the run; pages are cold, so impact on wall clock is small).
- **≥48 GB RAM box** — 8 workers is the sweet spot.

## Compatibility

- `Expander(app_config)` continues to work unchanged. The new
  `prebuilt_stock` parameter is opt-in for callers that build the
  shared block themselves.
- When `intern_cache=None` is passed (the default for direct
  `TreeMolecule(...)` construction), behavior falls back to the
  original eager construction path. All existing tests pass without
  modification.

## Tests

New:
- `tests/test_lru.py` — covers the bounded LRU primitive.
- `tests/test_shared_inchi_key_set.py` — including cross-process
  attach via `spawn` context.
- `tests/test_stock_query.py` — packed and shared query backends.

Run with:

```bash
conda run -n shallow-tree python -m unittest discover tests/ -v
```
