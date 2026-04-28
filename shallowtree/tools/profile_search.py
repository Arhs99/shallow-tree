"""Profiling script for shallow-tree search.

Usage:
    python -m shallowtree.tools.profile_search \
        --smiles-file shallowtree/smiles40.txt \
        --config config.yml \
        --depth 2 \
        --profile
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def timer(timers: dict, key: str):
    """Lightweight timer context manager for instrumenting code sections."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if key in timers:
        timers[key] += elapsed
    else:
        timers[key] = elapsed


def main():
    parser = argparse.ArgumentParser(description="Profile shallow-tree search")
    parser.add_argument("--smiles-file", required=True, help="Path to SMILES file (one per line)")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--depth", type=int, default=2, help="Search depth")
    parser.add_argument("--profile", action="store_true", help="Save .prof file for snakeviz")
    args = parser.parse_args()

    from shallowtree.interfaces.full_tree_search import Expander

    smiles = Path(args.smiles_file).read_text().strip().splitlines()
    smiles = [s.strip() for s in smiles if s.strip()]

    print(f"Loaded {len(smiles)} SMILES from {args.smiles_file}")

    expander = Expander(configfile=args.config)
    expander.expansion_policy.select_first()
    expander.filter_policy.select_first()
    expander.stock.select_first()
    expander._profiling = True

    if args.profile:
        prof = cProfile.Profile()
        prof.enable()

    wall_start = time.perf_counter()
    df = expander.search_tree(smiles, max_depth=args.depth)
    wall_elapsed = time.perf_counter() - wall_start

    if args.profile:
        prof.disable()
        prof_path = "profile_search.prof"
        prof.dump_stats(prof_path)
        print(f"\ncProfile data saved to {prof_path} (view with: snakeviz {prof_path})")
        print("\nTop 20 cumulative time functions:")
        stats = pstats.Stats(prof)
        stats.sort_stats("cumulative")
        stats.print_stats(20)

    # Print summary table
    timers = expander._timers
    total = wall_elapsed

    print(f"\n{'='*60}")
    print(f"PROFILING SUMMARY")
    print(f"{'='*60}")
    print(f"Total wall time:      {total:.3f}s")
    print(f"SMILES processed:     {len(smiles)}")
    print(f"Avg per molecule:     {total/len(smiles):.3f}s")
    print(f"{'='*60}")
    print(f"{'Section':<25} {'Time (s)':>10} {'% Total':>10}")
    print(f"{'-'*25} {'-'*10} {'-'*10}")

    for key in ["stock_check", "cache_lookup", "expansion_policy", "filter_batch", "recursive_dfs"]:
        t = timers.get(key, 0.0)
        pct = (t / total * 100) if total > 0 else 0
        print(f"{key:<25} {t:>10.3f} {pct:>9.1f}%")

    accounted = sum(timers.get(k, 0.0) for k in ["stock_check", "cache_lookup", "expansion_policy", "filter_batch", "recursive_dfs"])
    overhead = total - accounted
    pct = (overhead / total * 100) if total > 0 else 0
    print(f"{'other/overhead':<25} {overhead:>10.3f} {pct:>9.1f}%")

    print(f"\n{'='*60}")
    total_counter = timers.get("_counter_total", 0)
    total_cache = timers.get("_cache_counter_total", 0)
    cache_rate = (total_cache / total_counter * 100) if total_counter > 0 else 0
    print(f"Total DFS calls:      {total_counter}")
    print(f"Cache hits:           {total_cache}")
    print(f"Cache hit rate:       {cache_rate:.1f}%")

    solved = (df['score'] > 0.9).sum()
    print(f"Solved:               {solved}/{len(smiles)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
