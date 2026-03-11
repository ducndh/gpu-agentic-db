"""
Print ceiling analysis tables from experiment results.

Usage:
    python analysis/report.py results/results.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from tasks.tpch import TIER_MAP


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
        return tabulate(rows, headers=headers, tablefmt="github", floatfmt=".2f")
    lines = ["  ".join(str(h) for h in headers)]
    lines += ["  ".join(str(v) for v in row) for row in rows]
    return "\n".join(lines)


def print_summary(results: list[dict]):
    """Table 1: Overall ceiling by backend × model."""
    print("\n" + "="*90)
    print("TABLE 1: Colocation Ceiling by Model and Backend")
    print("  Data-path% = (fetch+serialize+tokenize) / total latency")
    print("  Ceiling speedup = total / (total - data_path)")
    print("="*90)

    rows = []
    for r in results:
        bd = r.get("stage_breakdown_ms", {})
        dp = round(bd.get("fetch", 0) + bd.get("serialize", 0) + bd.get("tokenize", 0), 2)
        llm = round(bd.get("llm_prefill", 0) + bd.get("llm_gen", 0), 2)
        rows.append([
            r["model"], r["backend"], r["scale_factor"], r["task"],
            r.get("n_turns", 1),
            round(r["total_ms"], 1),
            round(bd.get("sql_exec", 0), 1),
            dp, llm,
            r["data_path_pct"],
            r["colocation_speedup"],
            r.get("answer_correct", "N/A"),
        ])

    headers = [
        "model", "backend", "SF", "task", "turns",
        "total_ms", "sql_ms", "dp_ms", "llm_ms",
        "dp_%", "ceiling_x", "correct",
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
        def pct(key): return round(bd.get(key, 0) / total * 100, 2)
        rows.append([
            r["model"], r["backend"], r["scale_factor"], r["task"],
            pct("sql_exec"), pct("fetch"), pct("serialize"),
            pct("tokenize"), pct("llm_prefill"), pct("llm_gen"),
        ])

    headers = [
        "model", "backend", "SF", "task",
        "sql_%", "fetch_%", "ser_%", "tok_%", "prefill_%", "gen_%",
    ]
    print(_table(rows, headers))


def print_colocation_by_result_size(results: list[dict]):
    """Table 3: Colocation ceiling by result size tier."""
    print("\n" + "="*90)
    print("TABLE 3: Colocation Ceiling by Result Size Tier")
    print("="*90)

    by_tier: dict[str, list] = defaultdict(list)
    for r in results:
        tier = TIER_MAP.get(r["task"], "unknown")
        by_tier[tier].append(r)

    rows = []
    for tier in ["small", "medium", "large"]:
        tr = by_tier[tier]
        if not tr:
            continue
        avg_dp  = sum(r["data_path_pct"] for r in tr) / len(tr)
        avg_spd = sum(r["colocation_speedup"] for r in tr) / len(tr)
        max_spd = max(r["colocation_speedup"] for r in tr)
        max_dp  = max(r["data_path_pct"] for r in tr)
        rows.append([tier, len(tr), round(avg_dp, 2), round(max_dp, 2),
                     round(avg_spd, 3), round(max_spd, 3)])

    headers = ["tier", "n_runs", "avg_dp_%", "max_dp_%",
               "avg_ceiling_x", "max_ceiling_x"]
    print(_table(rows, headers))


def print_model_size_effect(results: list[dict]):
    """Table 4: How model size affects colocation ceiling."""
    print("\n" + "="*90)
    print("TABLE 4: Model Size Effect on Colocation Ceiling")
    print("  Faster inference → data-path is a LARGER fraction → colocation MORE valuable")
    print("="*90)

    by_model: dict[str, list] = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    rows = []
    for model, mrs in sorted(by_model.items()):
        n = len(mrs)
        avg_total  = sum(r["total_ms"] for r in mrs) / n
        avg_dp_pct = sum(r["data_path_pct"] for r in mrs) / n
        avg_spd    = sum(r["colocation_speedup"] for r in mrs) / n
        max_dp_pct = max(r["data_path_pct"] for r in mrs)
        bd_totals: dict = {}
        for r in mrs:
            for k, v in r.get("stage_breakdown_ms", {}).items():
                bd_totals[k] = bd_totals.get(k, 0) + v
        rows.append([
            model, n,
            round(avg_total, 0),
            round(bd_totals.get("llm_gen", 0) / n, 0),
            round(avg_dp_pct, 2),
            round(max_dp_pct, 2),
            round(avg_spd, 3),
        ])

    headers = ["model", "n", "avg_total_ms", "avg_gen_ms",
               "avg_dp_%", "max_dp_%", "avg_ceiling_x"]
    print(_table(rows, headers))


def print_sirius_vs_duckdb(results: list[dict]):
    """Table 5: SQL execution Sirius vs DuckDB."""
    print("\n" + "="*90)
    print("TABLE 5: Sirius GPU vs DuckDB CPU — SQL Execution Time")
    print("="*90)

    by_key: dict[tuple, dict] = {}
    for r in results:
        key = (r["model"], r["scale_factor"], r["task"])
        bd = r.get("stage_breakdown_ms", {})
        if key not in by_key:
            by_key[key] = {}
        by_key[key][r["backend"]] = bd.get("sql_exec", 0)

    rows = []
    for (model, sf, task), backends in sorted(by_key.items()):
        cpu_ms = backends.get("duckdb_cpu")
        gpu_ms = backends.get("sirius_gpu")
        if cpu_ms is not None and gpu_ms is not None and gpu_ms > 0:
            speedup = round(cpu_ms / gpu_ms, 2)
        else:
            speedup = "N/A"
        rows.append([model, sf, task,
                     round(cpu_ms, 1) if cpu_ms is not None else "N/A",
                     round(gpu_ms, 1) if gpu_ms is not None else "N/A",
                     speedup])

    if rows:
        headers = ["model", "SF", "task", "cpu_ms", "gpu_ms", "sirius_speedup"]
        print(_table(rows, headers))
    else:
        print("  (no paired CPU/GPU runs to compare)")


def print_incremental_ceiling(results: list[dict]):
    """Table 6: KV-cache-aware ceiling for multi-turn."""
    agentic = [r for r in results if r.get("n_turns", 1) > 1]
    if not agentic:
        agentic = results

    print("\n" + "="*90)
    print("TABLE 6: KV-Cache-Aware Colocation Ceiling (Multi-Turn)")
    print("  incr_dp_% = data_path / (data_path + incremental LLM per turn)")
    print("="*90)

    by_model: dict[str, list] = defaultdict(list)
    for r in agentic:
        by_model[r["model"]].append(r)

    rows = []
    for model, mrs in sorted(by_model.items()):
        avg_turns = sum(r.get("n_turns", 1) for r in mrs) / len(mrs)
        avg_dp    = sum(r["data_path_pct"] for r in mrs) / len(mrs)
        avg_idp   = sum(r.get("incr_data_path_pct", r["data_path_pct"]) for r in mrs) / len(mrs)
        max_idp   = max(r.get("incr_data_path_pct", r["data_path_pct"]) for r in mrs)
        rows.append([model, len(mrs), round(avg_turns, 1),
                     round(avg_dp, 2), round(avg_idp, 2), round(max_idp, 2)])

    headers = ["model", "n", "avg_turns",
               "naive_dp_%", "incr_dp_%", "max_incr_dp_%"]
    print(_table(rows, headers))


def print_task_pattern_analysis(results: list[dict]):
    """Table 7: Data-path by agentic pattern — shows which workflows benefit most."""
    print("\n" + "="*90)
    print("TABLE 7: Data-Path by Agentic Task Pattern")
    print("  Shows which agentic workflows benefit most from GPU colocation")
    print("="*90)

    # Categorize tasks by pattern
    PATTERN_MAP = {
        "discovery_revenue_1997": "Schema Discovery",
        "discovery_top_nations": "Schema Discovery",
        "multistep_top_customer_orders": "Multi-Step Dependent",
        "multistep_supplier_analysis": "Multi-Step Dependent",
        "ranked_suppliers_by_nation": "Complex Analytical",
        "monthly_trend_analysis": "Complex Analytical",
        "exploratory_segment_analysis": "Exploratory",
        "large_shipment_scan": "Large Result Set",
        "large_supplier_parts": "Large Result Set",
        "large_order_details": "Large Result Set",
    }

    by_pattern: dict[str, list] = defaultdict(list)
    for r in results:
        pattern = PATTERN_MAP.get(r["task"], "Other")
        by_pattern[pattern].append(r)

    rows = []
    for pattern in ["Schema Discovery", "Multi-Step Dependent", "Complex Analytical",
                     "Exploratory", "Large Result Set"]:
        pr = by_pattern.get(pattern, [])
        if not pr:
            continue
        n = len(pr)
        avg_turns = sum(r.get("n_turns", 1) for r in pr) / n
        avg_dp    = sum(r["data_path_pct"] for r in pr) / n
        max_dp    = max(r["data_path_pct"] for r in pr)
        avg_idp   = sum(r.get("incr_data_path_pct", r["data_path_pct"]) for r in pr) / n
        max_idp   = max(r.get("incr_data_path_pct", r["data_path_pct"]) for r in pr)
        avg_sql   = sum(r.get("sql_success", 0) for r in pr) / n
        rows.append([
            pattern, n, round(avg_turns, 1), round(avg_sql, 1),
            round(avg_dp, 2), round(max_dp, 2),
            round(avg_idp, 2), round(max_idp, 2),
        ])

    headers = ["pattern", "n", "avg_turns", "avg_sql_calls",
               "avg_dp_%", "max_dp_%", "avg_incr_dp_%", "max_incr_dp_%"]
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
    print_incremental_ceiling(results)
    print_task_pattern_analysis(results)

    # ── Verdict ──
    print("\n" + "="*90)
    print("COLOCATION VERDICT")
    print("="*90)

    avg_dp = sum(r["data_path_pct"] for r in results) / len(results)
    max_dp = max(r["data_path_pct"] for r in results)

    has_incr = any("incr_data_path_pct" in r for r in results)
    if has_incr:
        avg_idp = sum(r.get("incr_data_path_pct", r["data_path_pct"]) for r in results) / len(results)
        max_idp = max(r.get("incr_data_path_pct", r["data_path_pct"]) for r in results)
        print(f"Naive data-path %:      avg={avg_dp:.2f}%  max={max_dp:.2f}%")
        print(f"KV-cache-aware dp %:    avg={avg_idp:.2f}%  max={max_idp:.2f}%")
        avg_dp, max_dp = avg_idp, max_idp

    if max_dp < 5:
        verdict = "NOT WORTH IT — data path < 5% even in worst case."
    elif avg_dp < 5 and max_dp < 15:
        verdict = "MARGINAL — average < 5%, but up to {:.1f}% for large results.".format(max_dp)
    elif avg_dp >= 5 and max_dp >= 15:
        verdict = "WORTH INVESTIGATING — data path is {:.1f}% average, {:.1f}% peak.".format(avg_dp, max_dp)
    else:
        verdict = "MIXED — avg {:.1f}%, max {:.1f}%. Profile by result size.".format(avg_dp, max_dp)

    print(f"\nAverage data-path %: {avg_dp:.2f}%")
    print(f"Max data-path %:     {max_dp:.2f}%")
    print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    main()
