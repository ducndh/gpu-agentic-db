"""
Experiment C: Production-grade inference on L40S via vLLM.

Same agentic ReAct loop as Exp B, but with:
  - vLLM instead of llama.cpp (flash attention, prefix caching, real TTFT)
  - Larger models (7B BF16, 14B BF16, 32B AWQ) enabled by 48GB VRAM
  - Real prefill/decode split via RequestOutput.metrics (no estimation)
  - Prefix caching ON by default — simulates production multi-turn serving

This gives accurate data_path_pct under a modern inference stack, answering:
  "Is data path still negligible when LLM inference is 5-10x faster?"

Run on L40S:
    pip install vllm
    python experiments/exp_c_vllm_l40s.py --sf 1 --models qwen2.5-7b --tasks q6
    python experiments/exp_c_vllm_l40s.py          # full sweep
    python experiments/exp_c_vllm_l40s.py --no-prefix-cache   # compare w/o caching
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizers import Tokenizer

from core.agent import ReactAgent
from core.backends.duckdb_cpu import DuckDBCPUBackend
from core.backends.sirius_gpu import SiriusGPUBackend
from core.llm.vllm_backend import VLLMBackend
from tasks.tpch import TASKS

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"
VALIDATION_KEYS_FILE = RESULTS_DIR / "validation_keys.json"

# Models available on L40S (48GB VRAM)
# BF16 memory: 7B~15GB, 14B~29GB, 32B~65GB (use AWQ for 32B)
MODEL_CONFIGS = [
    {
        "name": "qwen2.5-7b-bf16",
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "dtype": "auto",
        "quantization": None,
        "max_model_len": 16384,
    },
    {
        "name": "qwen2.5-14b-bf16",
        "hf_id": "Qwen/Qwen2.5-14B-Instruct",
        "dtype": "auto",
        "quantization": None,
        "max_model_len": 16384,
    },
    {
        "name": "qwen2.5-32b-awq",
        "hf_id": "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "dtype": "auto",
        "quantization": "awq",
        "max_model_len": 8192,
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
    backend_name: str,
    sf: int,
    task,
    tokenizer: Tokenizer,
    validation_keys: dict,
) -> dict:
    dpath = db_path(sf)
    vkey = validation_keys.get(task.name, {}).get(str(sf))

    backend_cls = SiriusGPUBackend if backend_name == "sirius_gpu" else DuckDBCPUBackend

    with backend_cls(dpath) as backend:
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
            backend_name=backend_name,
            model_name=model_cfg["name"],
            scale_factor=sf,
            validation_key=vkey,
        )

    return record.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Experiment C: vLLM on L40S")
    parser.add_argument("--sf",       nargs="+", type=int, default=SCALE_FACTORS)
    parser.add_argument("--models",   nargs="+")
    parser.add_argument("--tasks",    nargs="+")
    parser.add_argument("--backends", nargs="+", default=BACKENDS, choices=BACKENDS)
    parser.add_argument("--output",   default=str(RESULTS_DIR / "exp_c_vllm_l40s.json"))
    parser.add_argument("--no-prefix-cache", action="store_true",
                        help="Disable prefix caching (shows cost without production optimization)")
    parser.add_argument("--gpu-mem",  type=float, default=0.90,
                        help="GPU memory utilization fraction for vLLM (default 0.90)")
    args = parser.parse_args()

    model_cfgs = MODEL_CONFIGS
    if args.models:
        model_cfgs = [m for m in MODEL_CONFIGS if m["name"] in args.models
                      or m["hf_id"].split("/")[-1].lower() in args.models]
    tasks_to_run = TASKS
    if args.tasks:
        tasks_to_run = [t for t in TASKS if t.name in args.tasks
                        or any(a in t.name for a in args.tasks)]

    RESULTS_DIR.mkdir(exist_ok=True)
    enable_prefix_caching = not args.no_prefix_cache

    print("Loading tokenizer (for data-path measurement only)...")
    try:
        tokenizer = Tokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        print("  Loaded Qwen2.5 tokenizer")
    except Exception:
        tokenizer = Tokenizer.from_pretrained("gpt2")
        print("  Loaded GPT-2 tokenizer (fallback)")

    validation_keys = load_validation_keys()
    print(f"  Validation keys: {len(validation_keys)} tasks")
    print(f"  Prefix caching: {'ON' if enable_prefix_caching else 'OFF'}")

    all_results = []
    total = len(model_cfgs) * len(args.backends) * len(args.sf) * len(tasks_to_run)
    run_n = 0

    for model_cfg in model_cfgs:
        print(f"\n{'='*70}")
        print(f"Loading model: {model_cfg['name']}  ({model_cfg['hf_id']})")
        print(f"  dtype={model_cfg['dtype']}  quantization={model_cfg['quantization']}")
        print(f"{'='*70}")

        llm = VLLMBackend(
            model=model_cfg["hf_id"],
            dtype=model_cfg["dtype"],
            quantization=model_cfg["quantization"],
            max_model_len=model_cfg["max_model_len"],
            gpu_memory_utilization=args.gpu_mem,
            enable_prefix_caching=enable_prefix_caching,
            temperature=0.0,
            max_tokens=512,
        )

        with llm:
            print("  Model loaded.")

            for backend_name in args.backends:
                for sf in args.sf:
                    if not Path(db_path(sf)).exists():
                        print(f"  [SKIP] DB not found: tpch_sf{sf}.duckdb")
                        continue
                    for task in tasks_to_run:
                        run_n += 1
                        print(f"\n  [{run_n}/{total}] {backend_name} | SF={sf} | {task.name}")
                        t0 = time.time()
                        try:
                            result = run_one(llm, model_cfg, backend_name, sf, task,
                                             tokenizer, validation_keys)
                            result["status"] = "ok"
                            result["prefix_caching"] = enable_prefix_caching
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            result = {
                                "task": task.name, "backend": backend_name,
                                "model": model_cfg["name"], "scale_factor": sf,
                                "mode": "agentic", "status": "error", "error": str(e),
                                "prefix_caching": enable_prefix_caching,
                            }
                        print(
                            f"    total={result.get('total_ms','?')}ms  "
                            f"turns={result.get('n_turns','?')}  "
                            f"data_path={result.get('data_path_pct','?')}%  "
                            f"incr_dp={result.get('incr_data_path_pct','?')}%  "
                            f"correct={result.get('answer_correct','?')}  "
                            f"retries={result.get('n_retries',0)}  "
                            f"wall={time.time()-t0:.1f}s"
                        )
                        all_results.append(result)
                        with open(args.output, "w") as f:
                            json.dump(all_results, f, indent=2)

    print(f"\nDone. {len(all_results)} runs saved to {args.output}")


if __name__ == "__main__":
    main()
