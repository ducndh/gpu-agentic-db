"""
GPU Colocation Ceiling Study — Unified Experiment Sweep

Runs the full ReAct agent loop across all combinations of:
  - Models (vLLM on GH200)
  - SQL backends (DuckDB CPU, Sirius GPU)
  - Scale factors (1, 5, 10)
  - Tasks (agentic TPC-H tasks)

Design decisions:
  - LLM loaded ONCE per model, reused across all combos
  - SQL backend kept alive per (backend, SF) pair — no repeated gpu_buffer_init
  - Fallback to gold SQL when LLM produces invalid SQL
  - Validation keys for retry-until-correct behavior
  - Results saved incrementally (resume-safe)

Run:
    # Smoke test
    python experiments/run_sweep.py --models Qwen/Qwen2.5-Coder-7B-Instruct --sf 1 --tasks discovery_revenue_1997

    # Full sweep
    python experiments/run_sweep.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizers import Tokenizer

from core.agent import ReactAgent
from core.backends.sql_backend import SQLBackend
from core.llm.vllm_backend import VLLMBackend
from tasks.tpch import TASKS, TASKS_BY_NAME

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"
VALIDATION_KEYS_FILE = RESULTS_DIR / "validation_keys.json"

# Models to sweep — each must fit on a single GH200 (96GB HBM3)
MODEL_CONFIGS = [
    {
        "name": "qwen2.5-coder-7b",
        "hf_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "dtype": "auto",
        "quantization": None,
        "max_model_len": 32768,
    },
    {
        "name": "qwen3.5-9b",
        "hf_id": "Qwen/Qwen3.5-9B",
        "dtype": "auto",
        "quantization": None,
        "max_model_len": 32768,
    },
    {
        "name": "qwen2.5-coder-32b",
        "hf_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "dtype": "auto",
        "quantization": None,
        "max_model_len": 32768,
    },
    {
        "name": "qwen3.5-27b",
        "hf_id": "Qwen/Qwen3.5-27B",
        "dtype": "auto",
        "quantization": None,
        "max_model_len": 32768,
    },
]

BACKENDS      = ["duckdb_cpu", "sirius_gpu"]
SCALE_FACTORS = [1, 5, 10]
N_WARMUP      = 2


def db_path(sf: int) -> str:
    return str(DATA_DIR / f"tpch_sf{sf}.duckdb")


def load_validation_keys() -> dict:
    if VALIDATION_KEYS_FILE.exists():
        with open(VALIDATION_KEYS_FILE) as f:
            return json.load(f)
    return {}


def run_one(
    llm: VLLMBackend,
    model_cfg: dict,
    backend: SQLBackend,
    task,
    sf: int,
    tokenizer: Tokenizer,
    validation_keys: dict,
) -> dict:
    """Run the full ReAct agent loop with a pre-loaded LLM and pre-connected backend."""
    vkey = validation_keys.get(task.name, {}).get(str(sf))

    # Warmup SQL caches
    for _ in range(N_WARMUP):
        backend.warmup(task.gold_sql)

    agent = ReactAgent(
        llm=llm,
        db=backend,
        tokenizer=tokenizer,
        max_turns=10,
        fallback_sql=task.gold_sql,
        max_retries=2,
    )

    record = agent.run(
        question=task.natural_question,
        schema_hint=task.schema_hint,
        task_name=task.name,
        backend_name=backend.name,
        model_name=model_cfg["name"],
        scale_factor=sf,
        validation_key=vkey,
    )

    return record.to_dict()


def main():
    parser = argparse.ArgumentParser(description="GPU Colocation Ceiling Study — Full Sweep")
    parser.add_argument("--sf",       nargs="+", type=int, default=SCALE_FACTORS)
    parser.add_argument("--models",   nargs="+", help="HuggingFace model IDs or short names")
    parser.add_argument("--tasks",    nargs="+", help="Task names to run")
    parser.add_argument("--backends", nargs="+", default=BACKENDS, choices=BACKENDS)
    parser.add_argument("--output",   default=str(RESULTS_DIR / "results.json"))
    parser.add_argument("--resume",   action="store_true",
                        help="Skip already-completed runs from output file")
    parser.add_argument("--gpu-mem",  type=float, default=0.85,
                        help="vLLM gpu_memory_utilization (reduce for Sirius colocation)")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Override max context length for all models")
    args = parser.parse_args()

    # Filter models
    model_cfgs = MODEL_CONFIGS
    if args.models:
        model_cfgs = [
            m for m in MODEL_CONFIGS
            if m["name"] in args.models or m["hf_id"] in args.models
        ]
        # Allow passing arbitrary HF IDs not in MODEL_CONFIGS
        known_ids = {m["hf_id"] for m in model_cfgs}
        for model_id in args.models:
            if model_id not in known_ids and "/" in model_id:
                model_cfgs.append({
                    "name": model_id.split("/")[-1].lower(),
                    "hf_id": model_id,
                    "dtype": "auto",
                    "quantization": None,
                    "max_model_len": args.max_model_len or 16384,
                })

    # Filter tasks
    tasks_to_run = TASKS
    if args.tasks:
        tasks_to_run = [t for t in TASKS if t.name in args.tasks]
        if not tasks_to_run:
            print(f"No matching tasks. Available: {[t.name for t in TASKS]}")
            sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)

    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = Tokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
        print("  Loaded Qwen2.5 tokenizer")
    except Exception:
        tokenizer = Tokenizer.from_pretrained("gpt2")
        print("  Loaded GPT-2 tokenizer (fallback)")

    # Load validation keys
    validation_keys = load_validation_keys()
    if validation_keys:
        print(f"  Loaded validation keys for {len(validation_keys)} tasks")

    # Resume support
    completed_keys: set = set()
    all_results = []
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            all_results = json.load(f)
        for r in all_results:
            completed_keys.add((r["model"], r["backend"], r["scale_factor"], r["task"]))
        print(f"  Resuming: {len(completed_keys)} runs already done")

    total = len(model_cfgs) * len(args.backends) * len(args.sf) * len(tasks_to_run)
    run_n = 0

    for model_cfg in model_cfgs:
        print(f"\n{'='*70}")
        print(f"Loading model: {model_cfg['hf_id']}")
        print(f"{'='*70}")

        max_model_len = args.max_model_len or model_cfg.get("max_model_len", 16384)
        llm = VLLMBackend(
            model=model_cfg["hf_id"],
            dtype=model_cfg.get("dtype", "auto"),
            quantization=model_cfg.get("quantization"),
            max_model_len=max_model_len,
            gpu_memory_utilization=args.gpu_mem,
            enable_prefix_caching=True,
            temperature=0.0,
            max_tokens=1024,
        )

        with llm:
            print(f"  Model loaded.")

            for backend_name in args.backends:
                use_sirius = (backend_name == "sirius_gpu")

                for sf in args.sf:
                    dpath = db_path(sf)
                    if not Path(dpath).exists():
                        print(f"  [SKIP] DB not found: tpch_sf{sf}.duckdb")
                        continue

                    # Keep backend alive across all tasks for this (backend, SF)
                    try:
                        backend = SQLBackend(dpath, use_sirius=use_sirius)
                        backend.connect()
                    except Exception as e:
                        print(f"  [SKIP] Backend {backend_name} failed: {e}")
                        continue

                    try:
                        for task in tasks_to_run:
                            run_n += 1
                            run_key = (model_cfg["name"], backend_name, sf, task.name)
                            if run_key in completed_keys:
                                print(f"\n  [{run_n}/{total}] SKIP (done): "
                                      f"{backend_name} | SF={sf} | {task.name}")
                                continue

                            print(f"\n  [{run_n}/{total}] {backend_name} | SF={sf} | {task.name}")
                            t0 = time.time()

                            try:
                                result = run_one(
                                    llm, model_cfg, backend, task, sf,
                                    tokenizer, validation_keys,
                                )
                                result["status"] = "ok"
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                result = {
                                    "task": task.name,
                                    "backend": backend_name,
                                    "model": model_cfg["name"],
                                    "scale_factor": sf,
                                    "mode": "agentic",
                                    "status": "error",
                                    "error": str(e),
                                }

                            correct = result.get("answer_correct", "?")
                            retries = result.get("n_retries", 0)
                            print(
                                f"    total={result.get('total_ms','?')}ms  "
                                f"turns={result.get('n_turns','?')}  "
                                f"data_path={result.get('data_path_pct','?')}%  "
                                f"incr_dp={result.get('incr_data_path_pct','?')}%  "
                                f"correct={correct}  retries={retries}  "
                                f"wall={time.time()-t0:.1f}s"
                            )

                            all_results.append(result)
                            # Save incrementally
                            with open(args.output, "w") as f:
                                json.dump(all_results, f, indent=2)
                    finally:
                        backend.close()

    print(f"\nDone. {len(all_results)} runs saved to {args.output}")


if __name__ == "__main__":
    main()
