"""
Experiment B: Fully agentic SQL generation.

The LLM receives the question and schema, generates SQL queries in a ReAct
loop, and arrives at a final answer.  All stages are timed.

This is the most realistic workflow simulation:
  User question → LLM generates SQL → DB executes → LLM reads result → ...

Sweep dimensions:
  - backend:      duckdb_cpu  |  sirius_gpu
  - scale_factor: 1 | 5 | 10
  - model:        all models in MODEL_CONFIGS
  - task:         all tasks in TASKS

Run:
    python experiments/exp_b_agentic.py [--sf 1] [--models qwen1.5b] [--tasks q6]
    python experiments/exp_b_agentic.py           # full sweep (long!)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizers import Tokenizer

from core.agent import ReactAgent
from core.backends.duckdb_cpu import DuckDBCPUBackend
from core.backends.sirius_gpu import SiriusGPUBackend
from core.llm.llama_backend import LlamaBackend
from tasks.tpch import TASKS

# ── Configuration ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"

MODEL_CONFIGS = [
    {
        "name": "qwen2.5-1.5b-q4_k_m",
        "path": str(MODELS_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        "n_ctx": 8192,
    },
    {
        "name": "qwen2.5-7b-q2_k",
        "path": str(MODELS_DIR / "qwen2.5-7b-instruct-q2_k.gguf"),
        "n_ctx": 8192,
    },
    {
        "name": "qwen2.5-7b-q4_k_m",
        # sharded — llama-cpp loads all shards when given the first
        "path": str(MODELS_DIR / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"),
        "n_ctx": 8192,
    },
    {
        "name": "qwen2.5-7b-q8_0",
        "path": str(MODELS_DIR / "qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf"),
        "n_ctx": 8192,
    },
    {
        "name": "qwen2.5-14b-q4_k_m",
        "path": str(MODELS_DIR / "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"),
        "n_ctx": 8192,
    },
]

BACKENDS = ["duckdb_cpu", "sirius_gpu"]
SCALE_FACTORS = [1, 5, 10]
WARMUP_RUNS = 1


def db_path(sf: int) -> str:
    return str(DATA_DIR / f"tpch_sf{sf}.duckdb")


def run_one(
    model_cfg: dict,
    backend_name: str,
    sf: int,
    task,
    tokenizer: Tokenizer,
) -> dict:
    """Run a single (model, backend, SF, task) combination and return result dict."""

    llm = LlamaBackend(
        model_path=model_cfg["path"],
        n_ctx=model_cfg["n_ctx"],
        n_gpu_layers=-1,
        temperature=0.0,
        max_tokens=512,
        verbose=False,
    )

    with llm:
        if backend_name == "sirius_gpu":
            backend = SiriusGPUBackend(db_path(sf))
        else:
            backend = DuckDBCPUBackend(db_path(sf))

        with backend:
            # Warmup: run gold SQL once to warm up the plan/data cache
            backend.warmup(task.gold_sql)

            agent = ReactAgent(
                llm=llm,
                db=backend,
                tokenizer=tokenizer,
                max_turns=5,
                fallback_sql=task.gold_sql,
            )

            record = agent.run(
                question=task.natural_question,
                schema_hint=task.schema_hint,
                task_name=task.name,
                backend_name=backend_name,
                model_name=model_cfg["name"],
                scale_factor=sf,
            )

    return record.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Experiment B: Fully agentic SQL")
    parser.add_argument("--sf",     nargs="+", type=int, default=SCALE_FACTORS,
                        help="Scale factor(s) to test, e.g. --sf 1 5")
    parser.add_argument("--models", nargs="+",
                        help="Model names to test (subset of MODEL_CONFIGS names)")
    parser.add_argument("--tasks",  nargs="+",
                        help="Task names to test (subset)")
    parser.add_argument("--backends", nargs="+", default=BACKENDS,
                        choices=BACKENDS)
    parser.add_argument("--output", default=str(RESULTS_DIR / "exp_b_agentic.json"))
    args = parser.parse_args()

    # Filter configs
    model_cfgs = MODEL_CONFIGS
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

    results = []
    total_runs = len(model_cfgs) * len(args.backends) * len(args.sf) * len(tasks_to_run)
    run_n = 0

    for model_cfg in model_cfgs:
        model_path = Path(model_cfg["path"])
        if not model_path.exists():
            print(f"  [SKIP] Model not found: {model_path.name}")
            continue

        print(f"\n{'='*70}")
        print(f"Model: {model_cfg['name']}")
        print(f"{'='*70}")

        for backend_name in args.backends:
            for sf in args.sf:
                dpath = db_path(sf)
                if not Path(dpath).exists():
                    print(f"  [SKIP] DB not found: {dpath} (run setup/generate_tpch.py first)")
                    continue

                for task in tasks_to_run:
                    run_n += 1
                    print(
                        f"\n[{run_n}/{total_runs}] "
                        f"{model_cfg['name']} | {backend_name} | SF={sf} | {task.name}"
                    )
                    t_start = time.time()
                    try:
                        result = run_one(model_cfg, backend_name, sf, task, tokenizer)
                        result["status"] = "ok"
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        result = {
                            "task": task.name,
                            "backend": backend_name,
                            "model": model_cfg["name"],
                            "scale_factor": sf,
                            "mode": "agentic",
                            "status": "error",
                            "error": str(e),
                        }
                    elapsed = time.time() - t_start
                    print(
                        f"  total={result.get('total_ms', '?')}ms  "
                        f"data_path={result.get('data_path_pct', '?')}%  "
                        f"ceiling_speedup={result.get('colocation_speedup', '?')}x  "
                        f"wall={elapsed:.1f}s"
                    )
                    results.append(result)

                    # Save incrementally
                    with open(args.output, "w") as f:
                        json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {args.output}")
    return results


if __name__ == "__main__":
    main()
