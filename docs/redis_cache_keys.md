# Redis Cache Key Design

shallow-tree caches search results in Redis so subtrees aren't recomputed across
molecules, processes, or runs. This document explains **how a cache key is formed** and,
crucially, **why it is namespaced by search mode and scaffold** — the property that keeps a
scaffold (context) search from poisoning a standard search.

> TL;DR — every key carries the search mode in its namespace, so the *same molecule* lives
> under *different keys* depending on how it was solved. A standard search can never read a
> scaffold search's context-only route.

---

## Anatomy of a key

All keys are built in a single chokepoint, `RedisCache._make_key`:

```python
return f"shallowtree:{self._config_hash}:{self._namespace}:{key_type}:{inchi_key}"
```

| Segment          | Source                                                          | Isolates…                          |
|------------------|----------------------------------------------------------------|------------------------------------|
| `shallowtree`    | literal prefix                                                 | this app's keyspace                |
| `{config_hash}`  | `sha256` of expansion/filter/stock **names** + cutoffs, `[:16]` | different model / stock configs    |
| `{namespace}`    | **search mode + scaffold** (see below)                         | standard ↔ scaffold, scaffold ↔ scaffold |
| `{key_type}`     | `cache` or `solved`                                            | pruning data vs route data         |
| `{inchi_key}`    | the molecule's InChIKey                                        | individual molecules               |

Two `key_type`s share the same namespace:

- **`cache`** → `(depth, score)` — branch-pruning data: "this molecule scores X when searched
  to this depth."
- **`solved`** → `(reactants, score, classification)` — route-reconstruction data: the chosen
  disconnection used to rebuild the tree and list building blocks.

---

## The namespace segment

The namespace is fixed once, when the `RedisCache` is constructed for a search instance
(`base_tree_search.py::_setup_redis_cache`). Because each search is a **single-mode instance**
(`StandardSearch` or `ScaffoldSearch`), the namespace never changes mid-run:

```python
scaffold = self._input_config.scaffold
if scaffold:                                    # ScaffoldSearch
    scaffold_hash = hashlib.sha256(scaffold.strip().encode()).hexdigest()[:16]
    namespace = f"scaffold:{scaffold_hash}"
else:                                           # StandardSearch
    namespace = "standard"
```

### Standard mode

```
namespace = "standard"

shallowtree:b2a1d2207f516873:standard:solved:MKRKLMOLSJUDAJ-UHFFFAOYSA-N
            └── config ──┘    └ mode ┘ └type┘ └──── InChIKey ─────────┘
```

### Scaffold mode

With scaffold `[*]c1n[nH]c2cc(-c3ccccc3)ccc12`:

```
namespace = "scaffold:a45c2805159a3b71"          # sha256(scaffold)[:16]

shallowtree:b2a1d2207f516873:scaffold:a45c2805159a3b71:solved:INJBORJJEYRVDE-UHFFFAOYSA-N
            └── config ──┘    └────────── mode ───────────┘ └type┘ └──── InChIKey ─────┘
```

> **Parsing note.** A scaffold namespace itself contains a colon (`scaffold:<hash>`), so a
> scaffold key has **six** segments to a standard key's **five**. `key_type` and `inchi_key`
> are always the last two — parse from the right.

---

## Why mode + scaffold belong in the key

A scaffold search deliberately accepts the **scaffold-matching reactant as a terminal node**
without expanding it — that's the whole point of a context search. But that terminal is only
"solved" *within that scaffold's context*; it may not be in stock at all.

Without a mode/scaffold namespace, that context-only `solved` entry is stored under the same
key a standard search would read:

```
            ┌─────────────────────── BEFORE (poisoning) ───────────────────────┐
scaffold run writes:   shallowtree:{cfg}:solved:{root}   →  route ends at a NON-STOCK terminal
standard run reads:    shallowtree:{cfg}:solved:{root}   →  replays it, emits non-stock "BB"  ✗
            └──────────────────────────────────────────────────────────────────┘

            ┌─────────────────────── AFTER (isolated) ─────────────────────────┐
scaffold run writes:   shallowtree:{cfg}:scaffold:{h}:solved:{root}
standard run reads:    shallowtree:{cfg}:standard:solved:{root}   →  miss → recompute clean  ✓
            └──────────────────────────────────────────────────────────────────┘
```

This was the root cause of intermittent `test_*_standard_search` failures: they passed only
against a freshly flushed Redis. The namespace closes it **by construction** — the keys simply
never overlap.

### What the namespace isolates

| Scenario                                  | Same key? | Result                                   |
|-------------------------------------------|-----------|------------------------------------------|
| standard ↔ standard (same config)         | yes       | shared — correct, both are stock-checked |
| standard ↔ scaffold                       | **no**    | isolated — no context leak               |
| scaffold S₁ ↔ scaffold S₂                 | **no**    | isolated — each scaffold is self-consistent |
| scaffold S ↔ scaffold S                   | yes       | shared — terminal is the legit terminal  |

A context terminal can therefore only reappear under a context search with the **same scaffold
and same config**, where it is exactly the intended terminal — never anywhere else.

---

## Notes & trade-offs

- **Scaffold hashing is string-based.** The hash comes from the raw scaffold string (stripped).
  Two different *spellings* of the same scaffold land in separate namespaces — a cache miss,
  not a correctness bug.
- **Config is keyed by stock *name*, not contents.** `config_hash` hashes the stock's name, so
  "same config" means same stock name. Swapping the file behind a name would still collide on
  `config_hash` — a separate axis from this fix.
- **Cost.** Sub-reactant standard solves computed *during* a scaffold run are written to the
  scaffold namespace, so a later standard search won't reuse them. This is a perf trade-off
  only; correctness is unaffected.

---

## Related code & tests

| What                         | Where                                                        |
|------------------------------|--------------------------------------------------------------|
| Key construction             | `shallowtree/context/cache/redis_cache.py` → `_make_key`     |
| Namespace selection          | `shallowtree/interfaces/search_modes/base_tree_search.py` → `_setup_redis_cache` |
| Config hashing               | `redis_cache.py` → `_compute_config_hash`                    |
| Isolation tests              | `tests/test_redis_cache.py` → `TestNamespaceIsolation`, `TestKeyFormat` |
