# Performance Optimization Plan for shallow-tree

## Background for Fresh Claude Sessions

shallow-tree is a retrosynthetic analysis tool for drug discovery. It predicts synthetic routes using depth-limited DFS. See `CLAUDE.md` for full project overview.

### Architecture (hot path)
```
searchcli (CLI entry) → Expander.search_tree() loops over SMILES
  → Expander.req_search_tree(mol, depth) [recursive DFS]
      1. Stock check: mol in self.stock → Stock.__contains__ → InMemoryInchiKeyQuery (set lookup on mol.inchi_key)
      2. Cache check: mol.inchi_key in self.cache (dict lookup)
      3. Expansion: self.expansion_policy.get_actions([mol]) → ML model predict → list of TemplatedRetroReaction
      4. Template application: action.reactants (lazy) → RDChiral/RDKit applies SMARTS template → TreeMolecule products
      5. Filter: self.filter_policy[name].batch_feasibility(ml_actions) → single batched ML predict
      6. Recurse: req_search_tree(reactant, depth+1) for each feasible reaction's products
```

### Key files
- `shallowtree/interfaces/full_tree_search.py` — `Expander` class, `search_tree()`, `req_search_tree()`, `context_search()`
- `shallowtree/context/policy/filter_strategies.py` — `QuickKerasFilter.batch_feasibility()` (already implemented)
- `shallowtree/context/policy/expansion_strategies.py` — `TemplateBasedExpansionStrategy.get_actions()`, `_update_cache()`
- `shallowtree/chem/reaction.py` — `TemplatedRetroReaction._apply_with_rdchiral()` (line 320), `_apply_with_rdkit()` (line 366), `_RdChiralProductWrapper` (line 595)
- `shallowtree/chem/mol.py` — `TreeMolecule`, `Molecule.inchi_key` (lazy, line 102), `fingerprint()` (cached, line 153)
- `shallowtree/utils/models.py` — `LocalKerasModel`, `LocalOnnxModel`, `ExternalModelViaGRPC`, `ExternalModelViaREST`
- `shallowtree/context/stock/stock.py` — `Stock.__contains__()` (line 65)
- `shallowtree/context/stock/queries.py` — `InMemoryInchiKeyQuery.__contains__()` (line 142)

### Already implemented optimizations (DO NOT redo)
- `batch_feasibility()` in `filter_strategies.py:151-200`
- Batched filter usage in `full_tree_search.py:212-218`
- Branch cache `self.cache` dict in `full_tree_search.py:177-181`
- Redis persistent cache in `full_tree_search.py:184-195`
- Molecule fingerprint caching in `mol.py:164-175`
- InChI key lazy evaluation in `mol.py:110-114`
- Expansion policy cache in `expansion_strategies.py:422-442`

---

## Phase 0 Profiling Results (2026-03-13) ✅ COMPLETE

**Baseline:** 40 SMILES, depth 2, local Keras models. **Total: 1247s (~31s/molecule).**

cProfile on `smiles40.txt` with `config.yml`, depth 2. 153,619 DFS calls, 467,857 template applications.

| Category | Time (s) | % of total | Key functions |
|---|---|---|---|
| **RDChiral template application** | ~530s | **42%** | `rdchiralRun` 176s, `initialize_rxn_from_smarts` 83s, `_RdChiralProductWrapper.__init__` 195s, `canonicalize_outcome_smiles` 46s |
| **TensorFlow/Keras inference** | ~241s | **19%** | `model.predict` 236s (includes ~130s TF framework overhead: data_adapter, structured_function, func_graph) |
| **TreeMolecule creation** | ~93s | **7%** | `TreeMolecule.__init__` 20s, `sanitize` 18s, `remove_atom_mapping` 21s |
| **Python/other overhead** | ~383s | **31%** | List/dict comprehensions, attribute access, recursion dispatch |

### Key finding: `_RdChiralProductWrapper.__init__` is 195s (16%)
This wraps the product molecule for RDChiral. Called 467,857 times (once per template application). The wrapper depends only on the molecule (not the template), so the same molecule produces the same wrapper every time. This is not addressed in the original plan — added as **Phase 2b**.

### Key finding: TF framework overhead is ~130s (10%)
Keras `model.predict()` spends ~130s in TF framework setup (`data_adapter`, `structured_function`, `func_graph_from_py_func`) on top of ~91s in actual inference. ONNX would eliminate this entirely.

### Key finding: `initialize_rxn_from_smarts` is 83s (7%)
Called 467,857 times — same templates recompiled for every molecule. With ~80 unique templates, caching eliminates >99% of these calls.

### Revised priority order (by expected impact)
1. **Phase 1** — Cross-molecule cache sharing (cuts 153k DFS calls and 467k template applications)
2. **Phase 2** — RDChiral template compilation cache (saves ~83s)
3. **Phase 2b** — `_RdChiralProductWrapper` cache (saves up to ~195s)
4. **Phase 6** — ONNX conversion (saves ~130s TF overhead + faster inference)
5. **Phase 3** — Stock cache + reorder (modest but easy)
6. **Phases 4-5** — gRPC/HTTP pooling (remote mode only)
7. ~~**Phase 7**~~ — **SKIP**: Cython not justified. Python overhead is spread across many small functions in RDChiral/RDKit internals, not concentrated in `req_search_tree` loop logic.

### Files created/modified
- Created `shallowtree/tools/profile_search.py` — profiling script with timer context manager
- Modified `shallowtree/interfaces/full_tree_search.py` — added `_profiling`, `_timers` to `Expander.__init__()`, timer guards in `req_search_tree()`
- Profiling data saved to `profile_search.prof` (viewable with `snakeviz`)

---

## Phase 1: Cross-Molecule Cache Sharing

**Status:** ✅ DONE

**Goal:** Avoid recomputing subtrees for intermediates shared across input molecules (common in lead optimization batches).

**Modify `shallowtree/interfaces/full_tree_search.py`:**

In `search_tree()` (lines 136-169):
- Move `self.cache = dict()` and `self.solved = dict()` **before** the for-loop (around line 144)
- Keep per-molecule: `self.BBs = []`, `self._counter`, `self._cache_counter`, `solution`
- Same change in `context_search()` (lines 67-134)

```python
def search_tree(self, smiles, max_depth=2):
    self.max_depth = max_depth
    self.cache = dict()       # Shared across all molecules
    self.solved = dict()      # Shared (needed by best_route for shared intermediates)
    rows = []
    for smi in smiles:
        solution = defaultdict(list)
        mol = TreeMolecule(parent=None, smiles=smi)
        self.BBs = []
        self._counter = 0
        self._cache_counter = 0
        if self.redis_cache:
            self._load_from_redis(mol)
        score = self.req_search_tree(mol, depth=0)
        ...
```

**Expected impact:** 1.2-3x on batches of structurally related molecules. Should dramatically reduce the 153,619 DFS calls and 467,857 template applications.

**Verify:** Run same 40-SMILES batch, sum `_cache_counter` across all molecules. Should increase significantly vs. before.

---

## Phase 2: RDChiral Template Compilation Cache

**Status:** ✅ DONE

**Goal:** Avoid recompiling the same SMARTS template for every molecule it's applied to.

**Modify `shallowtree/chem/reaction.py`:**

Add module-level cached functions:
```python
from functools import lru_cache

@lru_cache(maxsize=2048)
def _cached_rdchiral_reaction(smarts: str):
    return rdc.rdchiralReaction(smarts)

@lru_cache(maxsize=2048)
def _cached_rdkit_reaction(smarts: str):
    return AllChem.ReactionFromSmarts(smarts)
```

In `_apply_with_rdchiral()` (line 325):
- Change `reaction = rdc.rdchiralReaction(self.smarts)` to `reaction = _cached_rdchiral_reaction(self.smarts)`

In `_apply_with_rdkit()` (line 367):
- Change `rxn = AllChem.ReactionFromSmarts(self.smarts)` to `rxn = _cached_rdkit_reaction(self.smarts)`

**Safety note:** `rdchiralReaction` objects are read-only during `rdchiralRun` -- safe to reuse. For `AllChem.ReactionFromSmarts`, `RunReactants` is also safe (doesn't mutate the reaction). Verify with a quick test.

**Expected impact:** Saves ~83s (`initialize_rxn_from_smarts`). With ~80 unique templates across 467,857 calls, >99% are cache hits.

**Verify:** Log `_cached_rdchiral_reaction.cache_info()` after a run. Expect high hit ratio.

---

## Phase 2b: `_RdChiralProductWrapper` Cache

**Status:** ✅ DONE

**Goal:** Avoid redundantly wrapping the same product molecule for every template applied to it.

**Problem:** `_RdChiralProductWrapper.__init__()` costs 195s (16% of total). It is called once per template application (467,857 times), but the wrapper depends only on the molecule, not the template. When 80 templates are applied to the same molecule, the same wrapper is constructed 80 times.

**Modify `shallowtree/chem/reaction.py`:**

Add a module-level LRU cache keyed on the molecule's mapped SMILES (the wrapper input):
```python
@lru_cache(maxsize=4096)
def _cached_rdchiral_product_wrapper(mapped_smiles: str, mol: TreeMolecule):
    return _RdChiralProductWrapper(mol)
```

In `_apply_with_rdchiral()` (line 326), change:
```python
rct = _RdChiralProductWrapper(self.mol)
```
to:
```python
rct = _cached_rdchiral_product_wrapper(self.mol.mapped_smiles, self.mol)
```

**Safety note:** `_RdChiralProductWrapper` is read-only during `rdchiralRun` — the wrapper's attributes (`reactants`, `atoms_r`, `bonds_by_mapnum`, etc.) are not mutated. The `mapped_smiles` string is used as the hashable cache key since `TreeMolecule` is not hashable. Verify that `rdchiralRun` does not mutate the wrapper object.

**Caveat:** If `rdchiralRun` mutates the wrapper (e.g. modifies `self.reactants`), this cache would cause bugs. Must verify this before implementing. If mutation occurs, a shallow-copy approach could work instead: cache the expensive initialization, then `copy.copy()` per call.

**Expected impact:** Saves up to ~195s. With depth 2 and ~80 templates per molecule, each unique molecule's wrapper is built once instead of 80 times.

**Verify:** Log cache hit ratio. Count `_RdChiralProductWrapper.__init__` calls before/after.

---

## Phase 3: Stock Lookup Optimization

**Status:** ✅ DONE

**Goal:** Cache stock results in the DFS cache and fix minor anti-patterns.

**Modify `shallowtree/interfaces/full_tree_search.py`, `req_search_tree()`:**

1. **Cache stock hits** -- after `if mol in self.stock: return 1.0`, add:
   ```python
   if mol in self.stock:
       self.cache[mol.inchi_key] = (0, 1.0)  # Cache for future encounters
       return 1.0
   ```

2. **Fix `.keys()` anti-pattern** -- change `mol.inchi_key in self.cache.keys()` to `mol.inchi_key in self.cache` (avoids creating a view object on every call).

3. **Reorder for fast path** -- move cache check before stock check. Stock molecules will be cached after first encounter (from fix #1), so subsequent lookups skip the `Stock.__contains__` call entirely:
   ```python
   if mol.inchi_key in self.cache:
       self._cache_counter += 1
       cdepth, cscore = self.cache[mol.inchi_key]
       if cdepth <= depth:
           return cscore
   if mol in self.stock:
       self.cache[mol.inchi_key] = (0, 1.0)
       return 1.0
   ```

4. **Convert stock to `frozenset`** in `shallowtree/context/stock/queries.py`:
   - `InMemoryInchiKeyQuery.__init__()` currently stores `self._stock_inchikeys = set(inchis)` (lines 110, 122)
   - Change to `self._stock_inchikeys = frozenset(inchis)` — `frozenset.__contains__` is slightly faster due to immutability, and the stock never changes after load

**Also modify `shallowtree/context/stock/stock.py`:**
- Add `contains_inchi_key(self, inchi_key: str) -> bool` method for direct InChI key lookup without requiring a `Molecule` object (useful for future optimizations where InChI key is already known)

**Expected impact:** Modest per-molecule, but accumulates on batches with shared building blocks (very common -- the same stock chemicals appear in many routes). Combined with Phase 1 (shared cache), this means stock molecules are checked once total across the entire batch.

**Verify:** Count `Stock.__contains__` calls before/after on 40-SMILES batch.

---

## Phase 3b: Default to RDKit over RDChiral

**Status:** ✅ DONE

**Goal:** Skip RDChiral's chirality handling for queries without stereochemistry (the common case).

**Changes:**
- `shallowtree/chem/reaction.py:306` — `kwargs.get("use_rdchiral", True)` → `False`
- `shallowtree/context/policy/expansion_strategies.py:320` — `kwargs.get("use_rdchiral", True)` → `False`
- `shallowtree/context/policy/expansion_strategies.py:255` — `self.use_rdchiral = True` → `False`

Users can restore RDChiral with `use_rdchiral: true` in config.yml when chirality matters.

**Actual impact:** 805s → 528s (-277s). RDKit `_apply_with_rdkit` costs 245s vs RDChiral's 488s for the same templates. `rdchiralRun` and `_RdChiralProductWrapper.__init__` fully eliminated from the profile.

---

## Cumulative Profiling Results (40 SMILES, depth 2, local Keras)

| Metric | Baseline | After all phases | Change |
|--------|----------|-----------------|--------|
| **Wall time** | 1247s | 528s | **-719s (2.36x)** |
| **Per molecule** | 31.2s | 13.2s | **-18.0s** |
| **DFS calls** | 153,619 | 128,987 | -16% |
| **Template applications** | 467,857 | 382,694 | -18% |
| **RDChiral** | 933s | 0s | eliminated |
| **RDKit template application** | — | 245s | replaced RDChiral |
| **Keras predict** | 241s | 212s | -12% |
| **TreeMolecule.__init__** | 93s | 228s | +135s (RDKit produces more duplicates) |

### Current bottlenecks
1. **RDKit template application + TreeMolecule creation:** 245s + 228s (90%) — RDKit's `RunReactants` produces more duplicate outcomes than RDChiral
2. **Keras/TF inference:** 212s (40%) — TF framework overhead still ~110s; ONNX (Phase 6) would eliminate this

---

## Phase 4: gRPC Channel Pooling

**Status:** ✅ DONE

**Goal:** Reuse gRPC channels instead of creating one per prediction call.

**Modify `shallowtree/utils/models.py`, `ExternalModelViaGRPC`:**

In `__init__()` (lines 279-285):
```python
def __init__(self, key):
    ...
    self._sig_def = self._get_sig_def()  # Uses temp channel (existing behavior)
    # Create persistent channel + stub
    self._channel = grpc.insecure_channel(self._server)
    self._service = prediction_service_pb2_grpc.PredictionServiceStub(self._channel)
```

In `predict()` (lines 294-313):
- Remove lines 306-307 (channel/service creation per call)
- Use `self._service` and `self._channel` instead

Add cleanup:
```python
def close(self):
    if hasattr(self, '_channel'):
        self._channel.close()
```

**Expected impact:** Saves ~5-20ms per predict call. For remote mode with thousands of calls per batch, this is significant (10-60s saved on 40 molecules).

**Actual impact (80 SMILES, depth 2, TF Serving on RTX 4070 Ti, `parallel -j 8`):**
- Before: 1m 20.6s (1.01s/mol)
- After: 1m 17.4s (0.97s/mol) — ~4% faster, sub-1s per molecule

**GPU utilization note:** The RTX 4070 Ti sits at only 3-5% SM utilization during the run. The bottleneck is CPU-side (RDKit template application, TreeMolecule creation, Python overhead), not GPU inference or gRPC overhead. The GPU is starved for work — each `parallel` worker sends small sequential requests. Further GPU speedup would require batching expansion/filter predictions across molecules within each worker, but the DFS structure makes this difficult since each prediction depends on the previous one's results. The GPU is effectively a latency device here, not a throughput device.

---

## Phase 5: HTTP Session Pooling

**Status:** TODO

**Goal:** Reuse TCP connections for REST model backend.

**Modify `shallowtree/utils/models.py`, `ExternalModelViaREST`:**

In `__init__()` (line 201-206):
- Add `self._session = requests.Session()`

In `_handle_rest_api_request()` (line 240):
- Change `requests.request(...)` to `self._session.request(...)`

Add `close()` method: `self._session.close()`

**Expected impact:** ~3-10ms saved per REST call. Relevant when using REST backend.

**Verify:** Same as Phase 4 but with REST endpoint.

---

## Phase 6: ONNX Model Conversion

**Status:** TODO

**Goal:** Replace Keras/TF inference with ONNX Runtime for faster CPU inference.

**Create `shallowtree/tools/convert_to_onnx.py`:**
- Load Keras `.hdf5` model with custom objects (`top10_acc`, `top50_acc` from `models.py:99-107`)
- Convert via `tf2onnx.convert.from_keras()` with correct input signature (`(None, 2048)` for expansion, `(None, 2048), (None, 2048)` for filter)
- Save `.onnx` alongside original
- Validate: compare predictions on 100 random inputs, assert `np.allclose(keras_out, onnx_out, atol=1e-5)`

**Add `tf2onnx` to `pyproject.toml`** under optional extras.

**Usage:** After converting, change config model paths from `.hdf5` to `.onnx`. The existing `load_model()` in `models.py:74` already routes `.onnx` to `LocalOnnxModel`.

**Expected impact:** Saves ~130s TF framework overhead + faster raw inference. Profiling showed Keras spends 130s in `data_adapter`, `structured_function`, `func_graph_from_py_func` setup — none of which exists in ONNX Runtime.

**Verify:** Benchmark 40-SMILES batch with Keras vs ONNX. Verify identical scores (within tolerance).

---

## ~~Phase 7: Cython for `req_search_tree`~~ — SKIPPED

**Reason:** Profiling shows Python overhead is not concentrated in `req_search_tree` loop logic. The 31% "overhead" is spread across many small functions in RDChiral/RDKit internals (list comprehensions in `reaction.py`, dict comprehensions in `rdchiral/main.py`, etc.). Cython would not help these since they call into C extensions. The effort-to-impact ratio is poor.

---

## Summary

| Phase | Change | Files | Expected Savings | Effort | Status |
|-------|--------|-------|-----------------|--------|--------|
| 0 | Profiling script | `tools/profile_search.py`, `full_tree_search.py` | Baseline data | Low | ✅ Done |
| 1 | Cross-molecule cache | `full_tree_search.py` | -116s (1.10x) | Low | ✅ Done |
| 2 | RDChiral template cache | `reaction.py` | `initialize_rxn_from_smarts` eliminated | Low | ✅ Done |
| 2b | `_RdChiralProductWrapper` cache | `reaction.py` | Modest (effective after Phase 3) | Low | ✅ Done |
| 3 | Stock cache + reorder | `full_tree_search.py`, `queries.py` | -287s cumulative (1.55x) | Low | ✅ Done |
| 3b | Default to RDKit over RDChiral | `reaction.py`, `expansion_strategies.py` | -277s (1.55x → 2.36x) | Very Low | ✅ Done |
| 4 | gRPC channel pooling | `models.py` | 10-60s (remote only) | Low | ✅ Done |
| 5 | HTTP session pooling | `models.py` | Modest (REST only) | Very Low | TODO |
| 6 | ONNX conversion | `tools/convert_to_onnx.py`, `pyproject.toml` | ~130s+ TF overhead | Medium | TODO |
| ~~7~~ | ~~Cython~~ | — | — | — | SKIPPED |

**Achieved speedup (local mode):** 2.36x (1247s → 528s, 40 SMILES depth 2). Remaining bottlenecks: Keras/TF inference (212s, 40%), TreeMolecule creation (228s, 43%).

**Achieved speedup (remote mode, TF Serving + GPU):** 80 SMILES in 1m 17s (0.97s/mol) with `parallel -j 8`. GPU (RTX 4070 Ti) at 3-5% SM utilization — CPU-bound by RDKit/Python, not by inference or networking. The GPU serves as a low-latency inference endpoint but is far from saturated.

## Execution Approach

Execute phases sequentially. After completing each phase:
1. Update this plan to mark the phase as done and note any findings
2. Present results to the user
3. Ask whether to proceed to the next phase or skip it

The user can skip any phase at any time.

## Verification

After all phases:
1. Run profiling script on `smiles40.txt` with both local and remote configs
2. Run `pytest tests/ -v` to ensure no regressions
3. Compare scores for all 40 SMILES -- must be identical to baseline
4. Compare wall time to baseline
