"""
Print ceiling analysis tables from experiment results.

Usage:
    python analysis/report.py results/exp_a_fixed_sql.json
    python analysis/report.py results/exp_b_agentic.json
    python analysis/report.py results/exp_a_fixed_sql.json results/exp_b_agentic.json
"""

import json
import sys
from pathlib import Path

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


def load_results(paths: list[str]) -> list[dict]:
    results = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, list):
            results.extend(data)
        else:
            results.append(data)
    return [r for r in results if r.get("status") == "ok"]


def _table(rows, headers):
    if HAS_TABULATE:
        return tabulate(rows, headers=headers, tablefmt="github", floatfmt=".1f")
    # simple fallback
    lines = ["  ".join(str(h) for h in headers)]
    lines += ["  ".join(str(v) for v in row) for row in rows]
    return "\n".join(lines)


def print_summary(results: list[dict]):
    """Table 1: Overall ceiling by backend × model."""
    print("\n" + "="*90)
    print("TABLE 1: Colocation Ceiling by Model and Backend")
    print("  Data-path% = (fetch+serialize+tokenize) / total latency")
    print("  Ceiling speedup = total / (total - data_path)  [theoretical max from colocation]")
    print("="*90)

    rows = []
    for r in results:
        bd = r.get("stage_breakdown_ms", {})
        rows.append([
            r["model"],
            r["backend"],
            r["scale_factor"],
            r["task"],
            r["mode"],
            r["total_ms"],
            bd.get("sql_exec", 0),
            bd.get("fetch", 0) + bd.get("serialize", 0) + bd.get("tokenize", 0),
            bd.get("llm_prefill", 0) + bd.get("llm_gen", 0),
            r["data_path_pct"],
            r["colocation_speedup"],
        ])

    headers = [
        "model", "backend", "SF", "task", "mode",
        "total_ms", "sql_ms", "data_path_ms", "llm_ms",
        "data_path_%", "ceiling_x",
    ]
    print(_table(rows, headers))


def print_stage_breakdown(results: list[dict]):
    """Table 2: Per-stage breakdown as % of total."""
    print("\n" + "="*90)
    print("TABLE 2: Stage Breakdown (% of total latency)")
    print("="*90)

    rows = []
    for r in results:
        bd = r.get("stage_breakdown_ms", {})
        total = r["total_ms"] or 1
        def pct(key): return round(bd.get(key, 0) / total * 100, 1)
        rows.append([
            r["model"], r["backend"], r["scale_factor"], r["task"],
            pct("sql_exec"),
            pct("fetch"),
            pct("serialize"),
            pct("tokenize"),
            pct("llm_prefill"),
            pct("llm_gen"),
        ])

    headers = [
        "model", "backend", "SF", "task",
        "sql_%", "fetch_%", "serialize_%", "tokenize_%",
        "llm_prefill_%", "llm_gen_%",
    ]
    print(_table(rows, headers))


def print_colocation_by_result_size(results: list[dict]):
    """Table 3: Colocation ceiling grouped by expected result size tier."""
    print("\n" + "="*90)
    print("TABLE 3: Colocation Ceiling by Result Size Tier")
    print("  Shows how data-path overhead scales with query result size")
    print("="*90)

    # Group by task tier based on task name
    TIER_MAP = {
        "q6_discount_revenue":  "small",
        "q1_lineitem_summary":  "small",
        "q3_unshipped_revenue": "small",
        "q5_europe_revenue":    "small",
        "q16_supplier_count":   "medium",
        "large_shipment_scan":  "large",
    }

    from collections import defaultdict
    by_tier: dict[str, list] = defaultdict(list)
    for r in results:
        tier = TIER_MAP.get(r["task"], "unknown")
        by_tier[tier].append(r)

    rows = []
    for tier in ["small", "medium", "large"]:
        tier_results = by_tier[tier]
        if not tier_results:
            continue
        avg_dp_pct   = sum(r["data_path_pct"] for r in tier_results) / len(tier_results)
        avg_speedup  = sum(r["colocation_speedup"] for r in tier_results) / len(tier_results)
        max_speedup  = max(r["colocation_speedup"] for r in tier_results)
        rows.append([tier, len(tier_results), round(avg_dp_pct, 1),
                     round(avg_speedup, 3), round(max_speedup, 3)])

    headers = ["tier", "n_runs", "avg_data_path_%", "avg_ceiling_speedup", "max_ceiling_speedup"]
    print(_table(rows, headers))


def print_model_size_effect(results: list[dict]):
    """Table 4: How model size affects colocation ceiling."""
    print("\n" + "="*90)
    print("TABLE 4: Model Size Effect on Colocation Ceiling")
    print("  Larger/slower models → LLM dominates → smaller data-path %")
    print("="*90)

    from collections import defaultdict
    by_model: dict[str, list] = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    rows = []
    for model, model_results in sorted(by_model.items()):
        avg_total    = sum(r["total_ms"] for r in model_results) / len(model_results)
        avg_dp_pct   = sum(r["data_path_pct"] for r in model_results) / len(model_results)
        avg_speedup  = sum(r["colocation_speedup"] for r in model_results) / len(model_results)
        bd_totals    = {}
        for r in model_results:
            for k, v in r.get("stage_breakdown_ms", {}).items():
                bd_totals[k] = bd_totals.get(k, 0) + v
        n = len(model_results)
        rows.append([
            model, n,
            round(avg_total, 0),
            round(bd_totals.get("llm_gen", 0) / n, 0),
            round(avg_dp_pct, 1),
            round(avg_speedup, 3),
        ])

    headers = ["model", "n_runs", "avg_total_ms", "avg_llm_gen_ms",
               "avg_data_path_%", "avg_ceiling_speedup"]
    print(_table(rows, headers))


def print_sirius_vs_duckdb(results: list[dict]):
    """Table 5: SQL execution time Sirius vs DuckDB."""
    print("\n" + "="*90)
    print("TABLE 5: Sirius GPU vs DuckDB CPU — SQL Execution Time")
    print("="*90)

    from collections import defaultdict
    by_key: dict[tuple, dict] = {}
    for r in results:
        key = (r["model"], r["scale_factor"], r["task"])
        bd  = r.get("stage_breakdown_ms", {})
        if key not in by_key:
            by_key[key] = {}
        by_key[key][r["backend"]] = bd.get("sql_exec", 0)

    rows = []
    for (model, sf, task), backends in sorted(by_key.items()):
        cpu_ms = backends.get("duckdb_cpu", None)
        gpu_ms = backends.get("sirius_gpu", None)
        if cpu_ms and gpu_ms:
            speedup = round(cpu_ms / gpu_ms, 2) if gpu_ms > 0 else None
        else:
            speedup = None
        rows.append([model, sf, task,
                     round(cpu_ms, 1) if cpu_ms else "N/A",
                     round(gpu_ms, 1) if gpu_ms else "N/A",
                     speedup or "N/A"])

    headers = ["model", "SF", "task", "duckdb_cpu_ms", "sirius_gpu_ms", "sirius_speedup"]
    print(_table(rows, headers))


def main():
    if len(sys.argv) < 2:
        print("Usage: python analysis/report.py <results.json> [results2.json ...]")
        sys.exit(1)

    results = load_results(sys.argv[1:])
    if not results:
        print("No successful results found.")
        sys.exit(1)

    print(f"\nLoaded {len(results)} successful runs from {sys.argv[1:]}")

    print_summary(results)
    print_stage_breakdown(results)
    print_colocation_by_result_size(results)
    print_model_size_effect(results)
    print_sirius_vs_duckdb(results)

    print("\n" + "="*90)
    print("COLOCATION VERDICT")
    print("="*90)
    avg_dp = sum(r["data_path_pct"] for r in results) / len(results)
    max_dp = max(r["data_path_pct"] for r in results)
    if max_dp < 5:
        verdict = "NOT WORTH IT — data path < 5% even in worst case. LLM inference dominates."
    elif avg_dp < 5 and max_dp < 15:
        verdict = "MARGINAL — data path < 5% on average. Colocation helps only for large result sets."
    elif avg_dp >= 5 and max_dp >= 15:
        verdict = "WORTH INVESTIGATING — data path is a meaningful fraction for large results."
    else:
        verdict = f"MIXED — avg {avg_dp:.1f}%, max {max_dp:.1f}%. Profile by result size before investing."
    print(f"Average data-path %: {avg_dp:.1f}%")
    print(f"Max data-path %:     {max_dp:.1f}%")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
