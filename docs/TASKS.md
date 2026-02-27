# GPU Agentic DB — Investigation & Task Plan

## Context

This project measures whether GPU colocation (Sirius GPU-SQL + LLM on the same GPU) reduces latency in agentic database workflows. Current findings show data-path is <1% of total latency (LLM decode dominates). This plan expands the investigation across three areas: model selection, inference optimization, and experiment coverage — to gather numbers for a potential paper or production system.

Machine: NVIDIA L40S (48GB VRAM, Ada Lovelace, compute cap 8.9, CUDA 12.2)

---

## Area 1: Model Selection — Insights & Tasks

### Key Insights

1. **Current coverage is too narrow.** Only Qwen2.5 general-purpose instruct models are tested (1.5B, 7B, 14B, 32B-AWQ). This conflates "model quality" with "architecture choice" — we can't tell if the colocation ceiling is model-dependent or universal.

2. **Code-tuned variants likely dominate for text-to-SQL.** Qwen2.5-Coder models are trained on code/SQL data and consistently outperform their general-purpose counterparts on structured generation tasks. Adding Coder variants is the highest-value model addition.

3. **MoE models are interesting for latency.** DeepSeek-Coder-V2-Lite-Instruct is 15.7B total params but only ~2.4B active per token — inference speed closer to a 3B model with quality closer to 7B. If it fits in VRAM alongside Sirius, it could be a Pareto winner.

4. **The Pareto frontier is cheap to collect.** The experiment loop already sweeps all models automatically — just add entries to `MODEL_CONFIGS` in `exp_c_vllm_l40s.py`. Same tasks, same backends, only the model changes. Each model adds ~30 min to the full sweep.

5. **VRAM budget matters for colocation.** With `--gpu-mem 0.70` (needed for Sirius), larger models leave less KV cache. The 32B-AWQ model at ~20GB + Sirius 6GB + KV cache = tight on 48GB. FP8 quantization (WS2) unlocks larger models.

### Models to Add

| Priority | Model | HF ID | Size | VRAM (BF16) | Notes |
|----------|-------|-------|------|-------------|-------|
| P0 | Qwen2.5-Coder-1.5B | Qwen/Qwen2.5-Coder-1.5B-Instruct | 1.5B | ~3GB | Code-tuned small baseline |
| P0 | Qwen2.5-Coder-7B | Qwen/Qwen2.5-Coder-7B-Instruct | 7B | ~15GB | Likely best 7B for SQL |
| P1 | Llama-3.1-8B | meta-llama/Llama-3.1-8B-Instruct | 8B | ~16GB | May need HF token (gated) |
| P1 | Qwen2.5-Coder-32B-AWQ | Qwen/Qwen2.5-Coder-32B-Instruct-AWQ | 32B | ~20GB | Verify HF repo exists |
| P2 | DeepSeek-Coder-V2-Lite | deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct | 15.7B MoE | ~32GB | Tight VRAM, check chat template |
| P2 | Mistral-Small-24B | mistralai/Mistral-Small-24B-Instruct-2501 | 24B | ~48GB | Won't fit BF16, needs FP8/AWQ first |

### Implementation

**File:** `experiments/exp_c_vllm_l40s.py` — add to `MODEL_CONFIGS` list (line ~45)

Each entry follows the pattern:
```python
{"name": "qwen2.5-coder-7b-bf16", "hf_id": "Qwen/Qwen2.5-Coder-7B-Instruct", "dtype": "auto", "quantization": None, "max_model_len": 16384}
```

**Verification:** Run `--models <name> --sf 1 --tasks q6 --backends duckdb_cpu` smoke test for each new model. Check `correct=True` in output.

---

## Area 2: Inference Optimization — Insights & Tasks

### Key Insights

1. **Current setup already has the basics.** vLLM 0.16.0 with flash attention, prefix caching, CUDA graphs, chunked prefill — all enabled. The remaining wins are quantization, speculative decoding, engine choice, and backend lifecycle.

2. **FP8 is the lowest-hanging fruit.** L40S natively supports FP8 (Ada Lovelace). vLLM dynamically quantizes BF16→FP8 at load time — same HF model IDs, just add `quantization="fp8"`. Cuts memory ~2x with minimal accuracy loss. This also unlocks larger models (Mistral-24B, etc.).

3. **Speculative decoding is promising for SQL.** SQL tokens are highly predictable (keywords, table/column names from schema). A small draft model (Qwen2.5-0.5B) speculating for a larger target (Qwen2.5-7B) should have high acceptance rate. vLLM 0.16.0 supports `speculative_model` param. Could give 2-3x decode speedup. **Must verify it works in offline `LLM.generate()` mode** (not just the online server).

4. **SGLang vs vLLM for multi-turn.** Both handle prefix caching for growing conversation prefixes. SGLang's RadixAttention (radix-tree KV cache) is slightly more efficient for the exact pattern this agent uses — each turn sends full conversation history with a shared prefix. The real win is **lower per-request framework overhead** (~20-40% less overhead per LLM call). For a ReAct loop with 3-10 turns per task, that compounds. However, SGLang requires writing a new backend class matching the VLLMBackend interface.

5. **Sirius cold run is a major hidden cost.** The current code creates a **new Sirius subprocess per task** (`with backend_cls(dpath) as backend:` in `run_one()`). This means `gpu_buffer_init(4GB, 2GB)` — allocating 6GB of GPU memory — is paid for every single task, not once per experiment. The fix is simple: keep the backend alive across tasks that share the same DB file. The main loop nests as `for backend > for SF > for task` — move the `with backend:` to the SF level instead of inside `run_one()`.

6. **FP8 KV cache doubles capacity.** `kv_cache_dtype="fp8"` in vLLM stores KV pairs in FP8 instead of BF16. Doubles the number of tokens that fit in KV cache. Especially useful for multi-turn agents where context grows each turn.

### Task A: FP8 Model Configs (P0, trivial)

**File:** `experiments/exp_c_vllm_l40s.py`

Add 2 entries to MODEL_CONFIGS pointing to existing HF models with `quantization="fp8"`:
```python
{"name": "qwen2.5-7b-fp8",  "hf_id": "Qwen/Qwen2.5-7B-Instruct",  "dtype": "auto", "quantization": "fp8", "max_model_len": 16384}
{"name": "qwen2.5-14b-fp8", "hf_id": "Qwen/Qwen2.5-14B-Instruct", "dtype": "auto", "quantization": "fp8", "max_model_len": 16384}
```

The VLLMBackend already passes `quantization` through to vLLM at `vllm_backend.py:105-106`.

### Task B: FP8 KV Cache (P0, small)

**Files:** `core/llm/vllm_backend.py`, `experiments/exp_c_vllm_l40s.py`

1. Add `kv_cache_dtype` param to `VLLMBackend.__init__()` (line ~64)
2. Pass it to `LLM()` in `load()` (line ~96)
3. Add `--kv-cache-dtype` CLI flag in `exp_c_vllm_l40s.py` main()

### Task C: Speculative Decoding Investigation (P1, investigation first)

1. Check if vLLM 0.16.0 offline `LLM()` API accepts `speculative_model` param
2. If yes: add `speculative_model` and `num_speculative_tokens` to VLLMBackend
3. Draft model: `Qwen/Qwen2.5-0.5B-Instruct` (~1GB extra VRAM)
4. Expected benefit: 2-3x decode speedup for SQL generation (high token predictability)
5. Run same smoke test and compare wall time

### Task D: SGLang Backend (P2, medium effort)

**New file:** `core/llm/sglang_backend.py`

Must implement the same interface as VLLMBackend:
- `load()`, `unload()`, `chat(messages, prev_prompt_tokens) -> LLMResponse`, `__enter__`, `__exit__`
- Must return `LLMResponse` dataclass (defined in `core/llm/llama_backend.py:29-37`): `text`, `n_prompt_tokens`, `n_new_prompt_tokens`, `n_output_tokens`, `prefill_ms`, `prefill_incr_ms`, `decode_ms`, `total_ms`
- Must apply chat template (like `vllm_backend.py:143`)

Also need: new experiment script or CLI flag to switch engines.

**Key metric:** Compare TTFT on turns 2+ (where RadixAttention should outperform vLLM prefix caching).

### Task E: Sirius Warm-Pool Refactoring (P0, small)

**File:** `experiments/exp_c_vllm_l40s.py`

Current `run_one()` (line ~99): creates backend per task call.
Refactor main loop (line ~193) to create one backend per (backend_name, SF) pair:

```python
# BEFORE (current): backend created per task
for backend_name in args.backends:
    for sf in args.sf:
        for task in tasks_to_run:
            result = run_one(llm, model_cfg, backend_name, sf, task, ...)
            # run_one() internally does: with backend_cls(dpath) as backend:

# AFTER: backend kept alive across tasks
for backend_name in args.backends:
    for sf in args.sf:
        backend_cls = SiriusGPUBackend if backend_name == "sirius_gpu" else DuckDBCPUBackend
        with backend_cls(db_path(sf)) as backend:
            for _ in range(N_WARMUP):
                backend.warmup(tasks_to_run[0].gold_sql)
            for task in tasks_to_run:
                result = run_one(llm, model_cfg, backend, sf, task, ...)
```

This eliminates repeated `gpu_buffer_init()` calls (6GB GPU alloc per task → once per SF).

---

## Area 3: Experiment Coverage — Insights & Tasks

### Key Insights

1. **Current tasks are too uniform.** All 6 tasks are TPC-H analytical queries with full schema given upfront. All have a single gold SQL (even "2-step" tasks like q5 and q16 are really single queries with joins/subqueries). The agent is essentially a text-to-SQL translator, not a true agentic workflow.

2. **Real agentic DB workflows are messier.** In practice, users:
   - Don't know the schema — they explore first ("What tables do I have?")
   - Ask vague questions ("Anything interesting about 1997?")
   - Chain queries — step 2 depends on step 1's result ("Who's the top customer? Show their orders.")
   - Make mistakes and recover — the agent writes bad SQL, gets an error, fixes it
   - Use complex SQL patterns — window functions, CTEs, self-joins

3. **The validation system needs extending.** Current `validation_key` is a substring check in the final answer. For multi-step tasks, the "correct" answer may vary (e.g., exploratory tasks have no single right answer). Options: use loose validation keys (a distinctive value that should appear), or skip validation for exploratory tasks.

4. **These gaps matter for the paper.** If the colocation ceiling is different for multi-step vs single-step workflows (more data path per task due to more SQL calls), that's a finding worth reporting. Similarly, if error recovery adds extra turns (more LLM calls), the data_path_pct changes.

5. **Schema discovery is the most impactful gap.** In real deployments, the LLM spends turns querying `information_schema` or running `SHOW TABLES` before writing the actual analytical query. This adds 1-3 turns of pure overhead that the current benchmark doesn't capture.

### New Tasks to Add

All in `tasks/tpch.py`, following the existing `Task` dataclass.

#### Task 1: Schema Discovery (P0)

```python
Task(
    name="schema_discovery_revenue",
    natural_question=(
        "I have a database but I'm not sure what's in it. "
        "Can you explore the schema and then tell me the total revenue "
        "from orders in 1997? Revenue is extended price times (1 - discount)."
    ),
    schema_hint="You have access to a TPC-H database. Use SHOW TABLES and DESCRIBE to explore.",
    gold_sql="SELECT round(sum(l_extendedprice * (1 - l_discount)), 2) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey WHERE o.o_orderdate >= '1997-01-01' AND o.o_orderdate < '1998-01-01'",
    expected_result_tier="small",
    n_sql_steps=3,  # SHOW TABLES → DESCRIBE → actual query
)
```

Key: minimal `schema_hint` — forces the model to discover tables first.

#### Task 2: Multi-Step Dependent Query (P0)

```python
Task(
    name="multistep_top_customer_orders",
    natural_question=(
        "Find the customer who has spent the most money overall (by total order value). "
        "Then show me all of that customer's orders with their dates and total prices, "
        "sorted by date."
    ),
    schema_hint=TPCH_SCHEMA,
    gold_sql="SELECT o_orderkey, o_orderdate, o_totalprice FROM orders WHERE o_custkey = (SELECT o_custkey FROM orders GROUP BY o_custkey ORDER BY sum(o_totalprice) DESC LIMIT 1) ORDER BY o_orderdate",
    expected_result_tier="medium",
    n_sql_steps=2,  # find top customer → get their orders
)
```

#### Task 3: Error Recovery (P1)

```python
Task(
    name="error_recovery_revenue",
    natural_question=(
        "What is the total revenue by region for 1996? "
        "Revenue = extended_price * (1 - discount_rate)."
    ),
    schema_hint=TPCH_SCHEMA.replace("l_discount", "l_discount_rate"),  # deliberate misleading name
    gold_sql="SELECT r.r_name, round(sum(l.l_extendedprice * (1 - l.l_discount)), 2) AS revenue FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey JOIN customer c ON o.o_custkey = c.c_custkey JOIN nation n ON c.c_nationkey = n.n_nationkey JOIN region r ON n.n_regionkey = r.r_regionkey WHERE o.o_orderdate >= '1996-01-01' AND o.o_orderdate < '1997-01-01' GROUP BY r.r_name ORDER BY revenue DESC",
    expected_result_tier="small",
    n_sql_steps=2,  # first attempt fails, then corrects column name
)
```

Key: schema_hint has `l_discount_rate` instead of `l_discount` — first SQL attempt will fail, agent must recover from error message.

#### Task 4: Exploratory / Vague (P1)

```python
Task(
    name="exploratory_1997_patterns",
    natural_question=(
        "What are the most notable patterns or trends in the 1997 order data? "
        "Look at things like monthly trends, top segments, or any anomalies."
    ),
    schema_hint=TPCH_SCHEMA,
    gold_sql="SELECT date_trunc('month', o_orderdate) AS month, count(*) AS n_orders, round(sum(o_totalprice), 2) AS total_value FROM orders WHERE o_orderdate >= '1997-01-01' AND o_orderdate < '1998-01-01' GROUP BY month ORDER BY month",
    expected_result_tier="medium",
    n_sql_steps=3,  # exploratory: multiple angles
)
```

Validation: use loose key like "1997" — any reasonable analysis of 1997 data passes.

#### Task 5: Window Function / CTE (P1)

```python
Task(
    name="ranked_suppliers_by_nation",
    natural_question=(
        "Rank all suppliers by their total revenue within each nation. "
        "Show the top 3 suppliers per nation with their rank, name, nation, "
        "and total revenue. Use a window function."
    ),
    schema_hint=TPCH_SCHEMA,
    gold_sql="WITH supplier_rev AS (SELECT s.s_suppkey, s.s_name, n.n_name AS nation, round(sum(l.l_extendedprice * (1 - l.l_discount)), 2) AS revenue FROM lineitem l JOIN supplier s ON l.l_suppkey = s.s_suppkey JOIN nation n ON s.s_nationkey = n.n_nationkey GROUP BY s.s_suppkey, s.s_name, n.n_name) SELECT * FROM (SELECT *, row_number() OVER (PARTITION BY nation ORDER BY revenue DESC) AS rnk FROM supplier_rev) WHERE rnk <= 3 ORDER BY nation, rnk",
    expected_result_tier="medium",
    n_sql_steps=1,
)
```

### Post-Task Updates

After adding new tasks, update:
1. `setup/compute_validation_keys.py` — handle tasks with hardcoded `validation_key` (skip gold SQL execution for those)
2. `analysis/report.py` — add new task names to tier mapping so reporting groups them correctly

---

## Execution Priority

| Phase | Tasks | Effort | Value |
|-------|-------|--------|-------|
| 1 | Add Coder-1.5B + Coder-7B models, FP8 configs, Sirius warm-pool | Trivial | High — immediate data + perf win |
| 2 | Add 5 new experiment tasks, FP8 KV cache, Llama-3.1 model | Small-Medium | High — coverage + capacity |
| 3 | Speculative decoding investigation, remaining models (MoE, Mistral) | Medium | Medium — latency optimization |
| 4 | SGLang backend | Medium | Medium — alternative engine comparison |

---

## Environment Notes

- micromamba envs: `gpu-agentic-db` (Python 3.11, vLLM 0.16.0, duckdb 1.4.4), `sirius-worker` (duckdb 1.4.3)
- Sirius build: `~/sirius-dev/` branch `l40s-build`, extension at `build/release/extension/sirius/sirius.duckdb_extension`
- Run experiments: `SIRIUS_ROOT=~/sirius-dev micromamba run -n gpu-agentic-db python experiments/exp_c_vllm_l40s.py ...`
- Use `--gpu-mem 0.70` when running sirius_gpu backend (vLLM defaults to 0.90, leaves no room for Sirius's 6GB GPU buffers)
- All Qwen models are public (no HF token needed). Llama-3.1 may be gated.
