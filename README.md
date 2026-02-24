# GPU-Collocated Agentic Database: Ceiling Study

Measures whether running a GPU-accelerated SQL engine (Sirius) and an LLM
on the **same GPU** can meaningfully reduce end-to-end latency for agentic
database workflows.

## The Question

In a standard text-to-SQL agent pipeline, data flows:

```
GPU (SQL result) → CPU (fetchall) → CPU (serialize to text) → CPU (tokenize) → GPU (LLM KV cache)
```

The "data path" (fetch + serialize + tokenize) is pure overhead that
GPU colocation would eliminate. **Is this overhead large enough to matter?**

## Experiments

| Experiment | Description |
|---|---|
| `exp_a_fixed_sql.py` | Predefined (gold) SQL + LLM summarizes results. Cleanly isolates data-path overhead. |
| `exp_b_agentic.py`   | Fully agentic: LLM generates SQL in a ReAct loop. Realistic workflow simulation. |

Each experiment sweeps:
- **Backend**: DuckDB (CPU) vs Sirius (GPU)
- **Scale factor**: TPC-H SF=1, 5, 10
- **Model**: 5 configurations (size: 1.5B/7B/14B; quant: Q2/Q4/Q8)
- **Task**: 6 TPC-H analytical queries (small/medium/large result sets)

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Set up Sirius venv (duckdb 1.4.3 to match the Sirius extension)
bash setup/setup_sirius_venv.sh

# 3. Download GGUF models (~25 GB total)
python models/download.py

# 4. Generate TPC-H datasets (SF=1 takes ~5s, SF=10 takes ~5min)
python setup/generate_tpch.py
```

## Running

```bash
# Quick test: 1 model, 1 task, SF=1
python experiments/exp_a_fixed_sql.py --sf 1 --models qwen2.5-1.5b-q4_k_m --tasks q6_discount_revenue

# Full sweep (takes hours — run overnight)
python experiments/exp_a_fixed_sql.py
python experiments/exp_b_agentic.py

# Analyze results
python analysis/report.py results/exp_a_fixed_sql.json results/exp_b_agentic.json
```

## Key Metrics

- **`data_path_ms`**: Time spent on fetch + serialize + tokenize (what colocation eliminates)
- **`data_path_%`**: Data path as % of total latency
- **`colocation_speedup`**: Theoretical max speedup if data path → 0

## Architecture

```
core/
  timer.py            — Nanosecond stage timer, RunRecord/TurnRecord
  agent.py            — ReAct loop with per-stage instrumentation
  backends/
    duckdb_cpu.py     — DuckDB 1.4.4 in-process (standard pip package)
    sirius_gpu.py     — Sirius via long-lived worker subprocess (duckdb 1.4.3 venv)
  llm/
    llama_backend.py  — llama-cpp-python with prefill/decode time split
tasks/
  tpch.py             — 6 TPC-H tasks: natural questions + gold SQL
experiments/
  exp_a_fixed_sql.py  — Fixed SQL sweep
  exp_b_agentic.py    — Agentic SQL sweep
analysis/
  report.py           — Print ceiling tables
```

## Models Tested

| Model | Quant | VRAM | Purpose |
|---|---|---|---|
| Qwen2.5-1.5B | Q4_K_M | ~1 GB | Baseline (already downloaded) |
| Qwen2.5-7B   | Q2_K   | ~2.7 GB | Speed extreme |
| Qwen2.5-7B   | Q4_K_M | ~4.7 GB | Balanced |
| Qwen2.5-7B   | Q8_0   | ~8.1 GB | Near-full quality |
| Qwen2.5-14B  | Q4_K_M | ~9.0 GB | High quality |

Sirius GPU buffer: 4 GB cache + 2 GB proc = 6 GB.
All models coexist with Sirius on the 24 GB RTX 6000 except 14B Q4 (tight at 15 GB).
