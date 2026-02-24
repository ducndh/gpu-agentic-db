"""
Experiment A: Fixed SQL + LLM summarization.

The SQL queries are predefined (gold SQL from tasks.py). The LLM only
summarizes the result in natural language. This ISOLATES data-path
overhead from LLM SQL-generation quality.

Key design decisions:
  - LLM is loaded ONCE per model and reused across all backend/task combos
    (stable timings, no model-load variance, realistic for long-running agents)
  - SQL is executed N_WARMUP times to warm up the DB cache before timing
  - Each (backend, SF, task) is run N_ITER times; results are averaged

Sweep dimensions:
  - backend:      duckdb_cpu  |  sirius_gpu
  - scale_factor: 1 | 5 | 10
  - model:        all models in MODEL_CONFIGS
  - task:         all tasks in TASKS

Run:
    python experiments/exp_a_fixed_sql.py --sf 1 --models qwen2.5-1.5b-q4_k_m --tasks q6
    python experiments/exp_a_fixed_sql.py           # full sweep
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

# ── Config ────────────────────────────────────────────────────────────────────

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
N_WARMUP      = 2   # SQL warmup runs (not timed)
N_ITER        = 3   # Timed iterations to average

SUMMARIZE_PROMPT = """\
Here is the result of a SQL query run against a TPC-H database:

{result_table}

Write a concise 2-3 sentence summary of what this data shows.
Focus on the key insight.
"""


def measure_one_sql(
    backend,
    sql: str,
    tokenizer: Tokenizer,
    backend_name: str,
) -> TurnRecord:
    """
    Execute SQL + data path pipeline once.
    Returns a TurnRecord with sql_exec, fetch, serialize, tokenize stages.
    LLM stages are NOT included here — they're measured separately and added later.
    """
    turn = TurnRecord(turn_idx=0)

    try:
        qr: QueryResult = backend.execute(sql)
    except Exception as e:
        qr = QueryResult([], [], 0.0, 0.0, backend_name, str(e))

    turn.add("sql_exec", qr.exec_ms, n_rows=qr.n_rows, ok=qr.ok)
    turn.add("fetch",    qr.fetch_ms, n_rows=qr.n_rows)

    with StageTimer() as ser:
        result_text = format_result_as_markdown(qr.columns, qr.rows)
    turn.add("serialize", ser.elapsed_ms, n_bytes=len(result_text.encode()))

    with StageTimer() as tok:
        encoding = tokenizer.encode(result_text)
    turn.add("tokenize", tok.elapsed_ms, n_tokens=len(encoding.ids))

    # Store result_text for LLM call
    turn._result_text = result_text  # type: ignore[attr-defined]
    return turn


def run_one_fixed(
    llm: LlamaBackend,
    model_cfg: dict,
    backend_name: str,
    sf: int,
    task,
    tokenizer: Tokenizer,
) -> dict:
    """
    Run gold SQL N_ITER times, average stage timings, then summarize with LLM.
    Returns a result dict.
    """
    dpath = str(DATA_DIR / f"tpch_sf{sf}.duckdb")

    record = RunRecord(
        task_name=task.name,
        backend=backend_name,
        model_name=model_cfg["name"],
        scale_factor=sf,
        mode="fixed_sql",
    )

    if backend_name == "sirius_gpu":
        backend_cls = SiriusGPUBackend
    else:
        backend_cls = DuckDBCPUBackend

    with backend_cls(dpath) as backend:
        # Warmup: run gold SQL N_WARMUP times to populate DB/GPU caches
        for _ in range(N_WARMUP):
            backend.warmup(task.gold_sql)

        # Timed iterations for SQL + data path
        sql_turns = []
        for _ in range(N_ITER):
            sql_turns.append(
                measure_one_sql(backend, task.gold_sql, tokenizer, backend_name)
            )

        # Use the last iteration's result_text for the LLM call (most representative)
        result_text = getattr(sql_turns[-1], "_result_text", "(no result)")

    # Average SQL + data path timings across iterations
    avg_turn = TurnRecord(turn_idx=0)
    for stage in ("sql_exec", "fetch", "serialize", "tokenize"):
        vals = [t.get(stage) or 0.0 for t in sql_turns]
        avg_ms = sum(vals) / len(vals) if vals else 0.0
        # Collect metadata from last run
        last_meta = next(
            (r.metadata for t in sql_turns for r in t.records if r.stage == stage),
            {}
        )
        avg_turn.add(stage, avg_ms, **last_meta)

    record.sql_success_count = sum(
        1 for t in sql_turns
        for r in t.records if r.stage == "sql_exec" and r.metadata.get("ok", True)
    )

    # LLM summarization (1 call — already loaded LLM, no model-load overhead)
    prompt = SUMMARIZE_PROMPT.format(result_table=result_text)
    messages = [{"role": "user", "content": prompt}]
    llm_resp = llm.chat(messages)

    avg_turn.add("llm_prefill", llm_resp.prefill_ms,
                 n_prompt_tokens=llm_resp.n_prompt_tokens)
    avg_turn.add("llm_gen", llm_resp.decode_ms,
                 n_output_tokens=llm_resp.n_output_tokens,
                 tokens_per_sec=round(llm.tokens_per_second(llm_resp), 1))

    record.add_turn(avg_turn)
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

    model_cfgs   = MODEL_CONFIGS
    if args.models:
        model_cfgs = [m for m in MODEL_CONFIGS if m["name"] in args.models]
    tasks_to_run = TASKS
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

    all_results = []
    total = len(model_cfgs) * len(args.backends) * len(args.sf) * len(tasks_to_run)
    run_n = 0

    for model_cfg in model_cfgs:
        if not Path(model_cfg["path"]).exists():
            print(f"\n[SKIP] {model_cfg['name']}: model file not found")
            continue

        print(f"\n{'='*70}")
        print(f"Loading model: {model_cfg['name']}")
        print(f"{'='*70}")

        # Load LLM ONCE per model — reused across all backend/SF/task combos
        llm = LlamaBackend(
            model_path=model_cfg["path"],
            n_ctx=model_cfg["n_ctx"],
            n_gpu_layers=-1,
            temperature=0.0,
            max_tokens=256,
            verbose=False,
        )

        with llm:
            print(f"  Model loaded. Running {len(args.backends)} × {len(args.sf)} × {len(tasks_to_run)} combinations...")

            for backend_name in args.backends:
                for sf in args.sf:
                    dpath = DATA_DIR / f"tpch_sf{sf}.duckdb"
                    if not dpath.exists():
                        print(f"  [SKIP] DB not found: tpch_sf{sf}.duckdb")
                        continue
                    for task in tasks_to_run:
                        run_n += 1
                        print(
                            f"\n  [{run_n}/{total}] {backend_name} | SF={sf} | {task.name}"
                            f"  (avg of {N_ITER} SQL runs)"
                        )
                        t0 = time.time()
                        try:
                            result = run_one_fixed(llm, model_cfg, backend_name, sf, task, tokenizer)
                            result["status"] = "ok"
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            result = {
                                "task": task.name, "backend": backend_name,
                                "model": model_cfg["name"], "scale_factor": sf,
                                "mode": "fixed_sql", "status": "error", "error": str(e),
                            }
                        print(
                            f"    total={result.get('total_ms','?')}ms  "
                            f"data_path={result.get('data_path_pct','?')}%  "
                            f"ceiling={result.get('colocation_speedup','?')}x  "
                            f"wall={time.time()-t0:.1f}s"
                        )
                        all_results.append(result)
                        with open(args.output, "w") as f:
                            json.dump(all_results, f, indent=2)

    print(f"\nDone. {len(all_results)} runs saved to {args.output}")


if __name__ == "__main__":
    main()
