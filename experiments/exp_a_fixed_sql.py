"""
Experiment A: Fixed SQL + LLM summarization.

The SQL queries are predefined (gold SQL from tasks.py). The LLM only does
two things per turn:
  1. Summarizes the query result in natural language
  2. (Optionally) decides if it needs another query

This cleanly ISOLATES the data-path overhead from LLM SQL-generation quality:
  - Timing is NOT affected by whether the LLM writes good or bad SQL
  - Directly measures: fetch + serialize + tokenize as % of (SQL + LLM time)

Compare with Experiment B to understand:
  - How much of the latency is "LLM SQL writing quality" (B - A)
  - The pure colocation ceiling independent of SQL generation (from A)

Sweep dimensions: same as Experiment B
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizers import Tokenizer

from core.backends.duckdb_cpu import DuckDBCPUBackend, QueryResult
from core.backends.sirius_gpu import SiriusGPUBackend
from core.agent import format_result_as_markdown
from core.llm.llama_backend import LlamaBackend
from core.timer import RunRecord, StageTimer, TurnRecord
from tasks.tpch import TASKS

# ── Config (same paths as exp_b) ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"

MODEL_CONFIGS = [
    {"name": "qwen2.5-1.5b-q4_k_m",
     "path": str(MODELS_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"), "n_ctx": 8192},
    {"name": "qwen2.5-7b-q2_k",
     "path": str(MODELS_DIR / "qwen2.5-7b-instruct-q2_k.gguf"), "n_ctx": 8192},
    {"name": "qwen2.5-7b-q4_k_m",
     # sharded — llama-cpp loads all shards when given the first
     "path": str(MODELS_DIR / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"), "n_ctx": 8192},
    {"name": "qwen2.5-7b-q8_0",
     "path": str(MODELS_DIR / "qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf"), "n_ctx": 8192},
    {"name": "qwen2.5-14b-q4_k_m",
     "path": str(MODELS_DIR / "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"), "n_ctx": 8192},
]

BACKENDS      = ["duckdb_cpu", "sirius_gpu"]
SCALE_FACTORS = [1, 5, 10]

SUMMARIZE_PROMPT = """\
Here is the result of a SQL query run against a database:

{result_table}

Please write a concise 2-3 sentence summary of what this data shows.
Focus on the key insight, not every row.
"""


def run_one_fixed(
    model_cfg: dict,
    backend_name: str,
    sf: int,
    task,
    tokenizer: Tokenizer,
) -> dict:
    """Run gold SQL, then ask LLM to summarize. Time every stage."""

    llm = LlamaBackend(
        model_path=model_cfg["path"],
        n_ctx=model_cfg["n_ctx"],
        n_gpu_layers=-1,
        temperature=0.0,
        max_tokens=256,
        verbose=False,
    )

    record = RunRecord(
        task_name=task.name,
        backend=backend_name,
        model_name=model_cfg["name"],
        scale_factor=sf,
        mode="fixed_sql",
    )

    dpath = str(DATA_DIR / f"tpch_sf{sf}.duckdb")

    with llm:
        if backend_name == "sirius_gpu":
            backend = SiriusGPUBackend(dpath)
        else:
            backend = DuckDBCPUBackend(dpath)

        with backend:
            backend.warmup(task.gold_sql)

            turn = TurnRecord(turn_idx=0)

            # ── Execute gold SQL ───────────────────────────────────────────────
            try:
                qr: QueryResult = backend.execute(task.gold_sql)
            except Exception as e:
                qr = QueryResult([], [], 0.0, 0.0, backend_name, str(e))

            if qr.ok:
                record.sql_success_count += 1
            else:
                record.sql_failure_count += 1

            turn.add("sql_exec", qr.exec_ms, n_rows=qr.n_rows)
            turn.add("fetch",    qr.fetch_ms, n_rows=qr.n_rows)

            # ── Serialize ──────────────────────────────────────────────────────
            with StageTimer() as ser:
                result_text = format_result_as_markdown(qr.columns, qr.rows)
            turn.add("serialize", ser.elapsed_ms, n_bytes=len(result_text.encode()))

            # ── Tokenize ───────────────────────────────────────────────────────
            with StageTimer() as tok:
                encoding = tokenizer.encode(result_text)
            n_tokens = len(encoding.ids)
            turn.add("tokenize", tok.elapsed_ms, n_tokens=n_tokens)

            # ── LLM: summarize result ──────────────────────────────────────────
            prompt = SUMMARIZE_PROMPT.format(result_table=result_text)
            messages = [{"role": "user", "content": prompt}]
            llm_resp = llm.chat(messages)

            turn.add("llm_prefill", llm_resp.prefill_ms,
                     n_prompt_tokens=llm_resp.n_prompt_tokens)
            turn.add("llm_gen", llm_resp.decode_ms,
                     n_output_tokens=llm_resp.n_output_tokens)

            record.add_turn(turn)
            record.final_answer = llm_resp.text

    return record.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Experiment A: Fixed SQL + LLM summary")
    parser.add_argument("--sf",       nargs="+", type=int, default=SCALE_FACTORS)
    parser.add_argument("--models",   nargs="+")
    parser.add_argument("--tasks",    nargs="+")
    parser.add_argument("--backends", nargs="+", default=BACKENDS, choices=BACKENDS)
    parser.add_argument("--output",   default=str(RESULTS_DIR / "exp_a_fixed_sql.json"))
    args = parser.parse_args()

    model_cfgs    = MODEL_CONFIGS
    if args.models:
        model_cfgs = [m for m in MODEL_CONFIGS if m["name"] in args.models]
    tasks_to_run  = TASKS
    if args.tasks:
        tasks_to_run = [t for t in TASKS if t.name in args.tasks]

    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading tokenizer...")
    try:
        tokenizer = Tokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        print("  Loaded Qwen2.5 tokenizer")
    except Exception:
        tokenizer = Tokenizer.from_pretrained("gpt2")
        print("  Loaded GPT-2 tokenizer (fallback)")

    results  = []
    total    = len(model_cfgs) * len(args.backends) * len(args.sf) * len(tasks_to_run)
    run_n    = 0

    for model_cfg in model_cfgs:
        if not Path(model_cfg["path"]).exists():
            print(f"  [SKIP] {model_cfg['name']}: model file not found")
            continue

        print(f"\n{'='*70}")
        print(f"Model: {model_cfg['name']}")
        print(f"{'='*70}")

        for backend_name in args.backends:
            for sf in args.sf:
                dpath = DATA_DIR / f"tpch_sf{sf}.duckdb"
                if not dpath.exists():
                    print(f"  [SKIP] DB not found: tpch_sf{sf}.duckdb")
                    continue
                for task in tasks_to_run:
                    run_n += 1
                    print(f"\n[{run_n}/{total}] {model_cfg['name']} | {backend_name} | SF={sf} | {task.name}")
                    t0 = time.time()
                    try:
                        result = run_one_fixed(model_cfg, backend_name, sf, task, tokenizer)
                        result["status"] = "ok"
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        result = {
                            "task": task.name, "backend": backend_name,
                            "model": model_cfg["name"], "scale_factor": sf,
                            "mode": "fixed_sql", "status": "error", "error": str(e),
                        }
                    print(
                        f"  total={result.get('total_ms','?')}ms  "
                        f"data_path={result.get('data_path_pct','?')}%  "
                        f"ceiling_speedup={result.get('colocation_speedup','?')}x  "
                        f"wall={time.time()-t0:.1f}s"
                    )
                    results.append(result)
                    with open(args.output, "w") as f:
                        json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()
