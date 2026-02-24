# GPU Agentic DB — Colocation Ceiling Study

Measures whether GPU colocation (running Sirius GPU-SQL + LLM on the same GPU) meaningfully reduces latency in agentic database workflows.

**Verdict from current data:** data-path (fetch + serialize + tokenize) is **0.03–0.2% of total latency** across all model sizes and turn counts. LLM decode dominates. Colocation of the data-copy path is not worth engineering time on current hardware. The more interesting finding: Sirius shows 15–156× SQL speedups on large scans at SF=10, but SQL is <1% of total time so it doesn't move agent latency.

The open question: on production inference (L40S + vLLM), LLM time drops 5–10× while data-path stays pinned to CPU speed. Exp C measures this.

---

## The Question

In a standard agentic SQL pipeline, data flows:

```
GPU (SQL result) → CPU (fetchall) → CPU (serialize to text) → CPU (tokenize) → GPU (LLM prefill)
```

The "data path" (fetch + serialize + tokenize) is what GPU colocation would eliminate. **Is this overhead large enough to matter?**

---

## Setup

### This machine (RTX 6000, llama.cpp)

```bash
# 1. Create Python env and install deps
python -m venv .venv/main && source .venv/main/bin/activate
pip install -r requirements.txt

# 2. Create Sirius-compatible venv (needs duckdb==1.4.3, system has 1.4.4)
bash setup/setup_sirius_venv.sh

# 3. Generate TPC-H databases (SF 1, 5, 10 — takes ~3 min total)
python setup/generate_tpch.py

# 4. Download GGUF models into models/  (~25GB total)
python models/download.py

# 5. Compute validation keys (expected answer substrings per task/SF)
python setup/compute_validation_keys.py
```

### L40S machine (vLLM, production stack)

```bash
# Single script handles everything: venv, vllm, HF model downloads, TPC-H data
bash setup/setup_l40s.sh
```

---

## Running Experiments

### Exp A — Fixed SQL (clean timing baseline)
LLM summarizes a pre-executed gold SQL result. Isolates data-path overhead with no agent noise.

```bash
# Smoke test (fast, ~2 min)
python experiments/exp_a_fixed_sql.py --sf 1 --models qwen2.5-1.5b-q4_k_m --tasks q6_discount_revenue

# Full sweep: 5 models × 2 backends × 3 SFs × 6 tasks = 180 runs (~2–3 hrs)
python experiments/exp_a_fixed_sql.py
```

### Exp B — Agentic ReAct (realistic multi-turn)
LLM generates SQL in a ReAct loop (Thought → SQL → Observation → repeat → Answer).
Retries up to 2× when the answer is wrong (validated against `results/validation_keys.json`).

```bash
# Smoke test
python experiments/exp_b_agentic.py --sf 1 --models qwen2.5-7b-q2_k --tasks q3_unshipped_revenue --backends duckdb_cpu

# Full sweep (180 runs, ~4–8 hrs)
python experiments/exp_b_agentic.py

# Resume after a crash — skips already-completed runs
python experiments/exp_b_agentic.py --resume
```

Key flags: `--sf`, `--models`, `--tasks`, `--backends duckdb_cpu|sirius_gpu`

### Exp C — Production inference on L40S (vLLM)
Same ReAct loop as Exp B but with vLLM: real TTFT timing, flash attention, prefix caching, larger models (7B BF16 / 14B BF16 / 32B AWQ).

```bash
# Run on the L40S after setup_l40s.sh
python experiments/exp_c_vllm_l40s.py --models qwen2.5-7b-bf16 --sf 1 --backends duckdb_cpu

# Full sweep
python experiments/exp_c_vllm_l40s.py

# Compare prefix caching on vs off
python experiments/exp_c_vllm_l40s.py --output results/exp_c_cache_on.json
python experiments/exp_c_vllm_l40s.py --output results/exp_c_cache_off.json --no-prefix-cache
```

---

## Analysis

```bash
# Single experiment
python analysis/report.py results/exp_a_fixed_sql.json

# Combined A + B (recommended)
python analysis/report.py results/exp_a_fixed_sql.json results/exp_b_agentic.json

# Full comparison including L40S
python analysis/report.py results/exp_a_fixed_sql.json results/exp_b_agentic.json results/exp_c_vllm_l40s.json
```

The report prints 7 tables:
1. Ceiling speedup per model × backend × task
2. Stage breakdown % (sql / fetch / serialize / tokenize / prefill / decode)
3. Ceiling by result size tier (small / medium / large)
4. Model size effect on data-path fraction
5. Sirius GPU vs DuckDB CPU SQL execution time
6. KV-cache-aware incremental ceiling (correct metric for multi-turn agents)
7. Inference stack comparison: llama.cpp vs vLLM

---

## Project Structure

```
gpu-agentic-db/
├── core/
│   ├── agent.py              # ReAct loop, per-stage timing, retry-until-correct
│   ├── timer.py              # StageTimer / TurnRecord / RunRecord
│   └── backends/
│       ├── duckdb_cpu.py     # In-process DuckDB 1.4.4
│       └── sirius_gpu.py     # Sirius GPU via long-lived subprocess (duckdb==1.4.3 venv)
│   └── llm/
│       ├── llama_backend.py  # llama-cpp-python, GGUF, incremental prefill tracking
│       └── vllm_backend.py   # vLLM, BF16/AWQ, real TTFT timing, prefix caching
├── experiments/
│   ├── exp_a_fixed_sql.py    # Fixed gold SQL + LLM summarize
│   ├── exp_b_agentic.py      # Full ReAct agent sweep (llama.cpp)
│   └── exp_c_vllm_l40s.py   # Full ReAct agent sweep (vLLM, L40S)
├── tasks/tpch.py             # 6 TPC-H tasks: questions, schema hints, gold SQL
├── analysis/report.py        # Ceiling analysis — 7 tables + verdict
├── setup/
│   ├── generate_tpch.py      # Generates TPC-H .duckdb files via CALL dbgen(sf=N)
│   ├── compute_validation_keys.py  # Runs gold SQL, saves expected answer substrings
│   ├── setup_sirius_venv.sh  # .venv/sirius with duckdb==1.4.3
│   └── setup_l40s.sh         # .venv/l40s with vLLM + HF model downloads
├── models/
│   └── download.py           # Download GGUF models from HuggingFace
├── data/                     # TPC-H .duckdb files  (gitignored, ~5–15 GB)
└── results/                  # JSON result files     (gitignored except validation_keys.json)
```

---

## Key Metrics

| Metric | Definition |
|--------|-----------|
| `data_path_pct` | `(fetch + serialize + tokenize) / total × 100` — naive colocation ceiling |
| `incr_data_path_pct` | `data_path / (data_path + incr_llm_per_turn) × 100` — KV-cache-aware ceiling for multi-turn |
| `colocation_speedup` | `total / (total - data_path)` — max theoretical speedup from colocation |
| `llm_prefill_incr` | Prefill estimated from new tokens only (not full context), approximates KV cache reuse |

**data_path** = fetch + serialize + tokenize = the CPU round-trip colocation would eliminate.

With llama.cpp, `prefill_ms` is estimated using a `PREFILL_RELATIVE_SPEED=4.0` ratio.
With vLLM (Exp C), `prefill_ms` is the real TTFT from `RequestOutput.metrics`.

## Backends

**DuckDB CPU** (`duckdb_cpu`): in-process DuckDB 1.4.4.

**Sirius GPU** (`sirius_gpu`): long-lived Python subprocess using `.venv/sirius` (duckdb==1.4.3 + Sirius extension). Communicates JSON over stdin/stdout. Requires Sirius installed at `/home/cc/sirius/`. Runs are skipped gracefully if unavailable.

## Models Tested (llama.cpp, RTX 6000)

| Model | Quant | VRAM |
|-------|-------|------|
| Qwen2.5-1.5B | Q4_K_M | ~1 GB |
| Qwen2.5-7B | Q2_K | ~2.7 GB |
| Qwen2.5-7B | Q4_K_M | ~4.7 GB |
| Qwen2.5-7B | Q8_0 | ~8.1 GB |
| Qwen2.5-14B | Q4_K_M | ~9.0 GB |

## Models Available (vLLM, L40S 48 GB)

| Model | Format | VRAM |
|-------|--------|------|
| Qwen2.5-7B-Instruct | BF16 | ~15 GB |
| Qwen2.5-14B-Instruct | BF16 | ~29 GB |
| Qwen2.5-32B-Instruct | AWQ INT4 | ~20 GB |
