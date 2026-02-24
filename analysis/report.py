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
            round(bd.get("fetch", 0) + bd.get("serialize", 0) + bd.get("tokenize", 0), 2),
            round(bd.get("llm_prefill", 0) + bd.get("llm_gen", 0), 2),
            r["data_path_pct"],
            r["colocation_speedup"],
            r.get("answer_correct", "N/A"),
            r.get("n_retries", "N/A"),
        ])

    headers = [
        "model", "backend", "SF", "task", "mode",
        "total_ms", "sql_ms", "data_path_ms", "llm_ms",
        "data_path_%", "ceiling_x", "correct", "retries",
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


def print_incremental_ceiling(results: list[dict]):
    """Table 6: KV-cache-aware ceiling for multi-turn ReAct.

    In multi-turn agentic workflows, each extra turn's LLM cost (with KV cache)
    is only the incremental prefill of new observation tokens + decode.
    Data-path (fetch+serialize+tokenize) accumulates at the same rate.
    This table shows how the colocation ceiling changes when accounting for this.
    """
    # Only include agentic runs with >1 turn (single-turn has no KV cache benefit)
    agentic = [r for r in results if r.get("mode") == "agentic" and r.get("n_turns", 1) > 1]
    if not agentic:
        # Fall back to all runs if no multi-turn data
        agentic = results

    print("\n" + "="*90)
    print("TABLE 6: KV-Cache-Aware Colocation Ceiling (Multi-Turn ReAct)")
    print("  incr_data_path_% = data_path / (data_path + incremental LLM per turn)")
    print("  With KV cache: each turn's LLM work ≈ process new observation tokens + decode")
    print("  This is the accurate ceiling for a real agentic workflow vs single-turn exp_a")
    print("="*90)

    from collections import defaultdict
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
                     round(avg_dp, 1), round(avg_idp, 1), round(max_idp, 1)])

    headers = ["model", "n_runs", "avg_turns",
               "naive_data_path_%", "incr_data_path_%", "max_incr_data_path_%"]
    print(_table(rows, headers))
    print()
    print("  naive_data_path_%  = data_path / total  (exp_a metric, under-counts colocation value)")
    print("  incr_data_path_%   = data_path / (data_path + incr_LLM)  (correct for multi-turn)")


def print_inference_stack_comparison(results: list[dict]):
    """Table 7: llama.cpp vs vLLM — how inference speed affects data-path fraction."""
    # Detect which stack each result came from based on mode/model name heuristics
    # vLLM runs have model names like "qwen2.5-7b-bf16", llama.cpp use "qwen2.5-7b-q4_k_m"
    vllm_runs  = [r for r in results if "bf16" in r.get("model","") or "awq" in r.get("model","")]
    llama_runs = [r for r in results if any(q in r.get("model","")
                  for q in ("q2_k","q4_k_m","q8_0","q4_km","q2k","q8"))]
    if not vllm_runs or not llama_runs:
        return  # Only print when both stacks are present

    print("\n" + "="*90)
    print("TABLE 7: Inference Stack Comparison — llama.cpp vs vLLM")
    print("  Shows how data-path fraction changes as LLM inference gets faster")
    print("="*90)

    def summarise(runs, label):
        if not runs:
            return []
        avg_total   = sum(r["total_ms"] for r in runs) / len(runs)
        avg_dp      = sum(r["data_path_pct"] for r in runs) / len(runs)
        avg_idp     = sum(r.get("incr_data_path_pct", r["data_path_pct"]) for r in runs) / len(runs)
        avg_turns   = sum(r.get("n_turns", 1) for r in runs) / len(runs)
        bd_totals: dict = {}
        for r in runs:
            for k, v in r.get("stage_breakdown_ms", {}).items():
                bd_totals[k] = bd_totals.get(k, 0) + v
        n = len(runs)
        avg_decode  = bd_totals.get("llm_gen", 0) / n
        avg_prefill = bd_totals.get("llm_prefill", 0) / n
        return [label, n, round(avg_turns, 1), round(avg_total, 0),
                round(avg_prefill, 0), round(avg_decode, 0),
                round(avg_dp, 2), round(avg_idp, 1)]

    rows = [
        summarise(llama_runs, "llama.cpp (GGUF quant)"),
        summarise(vllm_runs,  "vLLM (BF16/AWQ)"),
    ]
    rows = [r for r in rows if r]
    headers = ["stack", "n_runs", "avg_turns", "avg_total_ms",
               "avg_prefill_ms", "avg_decode_ms",
               "naive_dp_%", "incr_dp_%"]
    print(_table(rows, headers))
    if len(rows) == 2:
        speedup = rows[0][3] / rows[1][3] if rows[1][3] > 0 else 0
        dp_ratio = rows[1][7] / rows[0][7] if rows[0][7] > 0 else 0
        print(f"\n  vLLM is {speedup:.1f}x faster overall")
        print(f"  Data-path fraction is {dp_ratio:.1f}x larger with vLLM (relatively more important)")


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
    print_inference_stack_comparison(results)

    print("\n" + "="*90)
    print("COLOCATION VERDICT")
    print("="*90)
    avg_dp = sum(r["data_path_pct"] for r in results) / len(results)
    max_dp = max(r["data_path_pct"] for r in results)
    # Use incremental metric if available (multi-turn agentic runs)
    has_incr = any("incr_data_path_pct" in r for r in results)
    if has_incr:
        avg_idp = sum(r.get("incr_data_path_pct", r["data_path_pct"]) for r in results) / len(results)
        max_idp = max(r.get("incr_data_path_pct", r["data_path_pct"]) for r in results)
        print(f"Naive data-path % (total LLM denominator): avg={avg_dp:.1f}%  max={max_dp:.1f}%")
        print(f"KV-cache incr data-path % (per-turn):      avg={avg_idp:.1f}%  max={max_idp:.1f}%")
        avg_dp, max_dp = avg_idp, max_idp
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
