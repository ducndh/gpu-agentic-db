"""
ReAct (Reason + Act) agent loop with per-stage timing instrumentation.

The agent follows the standard ReAct pattern:
  Thought → Action (SQL) → Observation (result) → Thought → ... → Answer

Each turn is a complete Thought+Action+Observation cycle. We time every
stage precisely so we can compute the colocation ceiling.

Stage timing per turn:
  llm_gen     - LLM generates Thought + SQL action (decode)
  llm_prefill - LLM processes context with SQL result (prefill, estimated)
  sql_exec    - DB executes the SQL
  fetch       - fetchall() copies results to Python
  serialize   - Format result table as markdown (CPU work)
  tokenize    - Tokenize the formatted text (CPU work)

The data path = fetch + serialize + tokenize is what GPU colocation removes.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from tokenizers import Tokenizer

from core.backends.duckdb_cpu import QueryResult
from core.llm.llama_backend import LlamaBackend, LLMResponse
from core.timer import RunRecord, StageTimer, TurnRecord

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are a data analyst assistant with access to a SQL database.
To answer the user's question, you can run SQL queries using this format:

Action: SQL
```sql
SELECT ...
```

After seeing the query result (Observation), continue reasoning until you
have a final answer. When done, write:

Answer: <your final answer here>

Rules:
- Write exactly one SQL block per Action, then stop and wait for the result.
- The SQL must be valid DuckDB SQL.
- Do NOT use EXPLAIN or any DDL (CREATE, INSERT, DROP).
- Keep SQL concise; avoid SELECT * on large tables.
- If the result is empty or unexpected, try a different approach.
- Limit large scans to at most 1000 rows unless the user asks otherwise.

{schema}
"""

# ── Result formatting ─────────────────────────────────────────────────────────

def format_result_as_markdown(columns: list[str], rows: list[tuple]) -> str:
    """Convert query result to a markdown table string."""
    if not rows:
        return "(empty result)"
    MAX_ROWS = 100  # truncate very large results to keep context manageable
    lines = []
    lines.append("| " + " | ".join(str(c) for c in columns) + " |")
    lines.append("|" + "|".join(" --- " for _ in columns) + "|")
    for row in rows[:MAX_ROWS]:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    if len(rows) > MAX_ROWS:
        lines.append(f"*... {len(rows) - MAX_ROWS} more rows truncated*")
    return "\n".join(lines)


# ── SQL extraction ────────────────────────────────────────────────────────────

_SQL_BLOCK = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_ANSWER = re.compile(r"^Answer\s*:\s*(.+)", re.MULTILINE | re.DOTALL)


def extract_sql(text: str) -> Optional[str]:
    """Extract SQL from a markdown code block in the LLM output."""
    m = _SQL_BLOCK.search(text)
    return m.group(1).strip() if m else None


def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer if the LLM declared one."""
    m = _ANSWER.search(text)
    return m.group(1).strip() if m else None


# ── Agent ─────────────────────────────────────────────────────────────────────

class ReactAgent:
    """
    ReAct agent with per-stage timing.

    Parameters
    ----------
    llm: LlamaBackend
        Loaded LLM backend.
    db: DuckDBCPUBackend | SiriusGPUBackend
        Database backend (must already be connected).
    tokenizer: Tokenizer
        HuggingFace tokenizer (for measuring tokenization time).
    max_turns: int
        Maximum number of (Thought+Action+Observation) cycles before forcing
        a fallback answer.
    fallback_sql: str | None
        If provided, used when the LLM generates invalid SQL (fall back
        silently, keeps timing clean).
    """

    def __init__(
        self,
        llm: LlamaBackend,
        db,
        tokenizer: Tokenizer,
        max_turns: int = 5,
        fallback_sql: Optional[str] = None,
    ):
        self.llm = llm
        self.db = db
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.fallback_sql = fallback_sql

    def run(
        self,
        question: str,
        schema_hint: str,
        task_name: str,
        backend_name: str,
        model_name: str,
        scale_factor: int,
    ) -> RunRecord:
        """
        Run the agent on a question and return a fully-instrumented RunRecord.
        """
        record = RunRecord(
            task_name=task_name,
            backend=backend_name,
            model_name=model_name,
            scale_factor=scale_factor,
            mode="agentic",
        )

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(schema=schema_hint)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ]

        for turn_idx in range(self.max_turns):
            turn = TurnRecord(turn_idx=turn_idx)

            # ── LLM generates Thought + Action ────────────────────────────────
            with StageTimer() as llm_timer:
                llm_resp = self.llm.chat(messages)
            # We split into prefill/decode per LlamaBackend estimates
            turn.add("llm_prefill", llm_resp.prefill_ms,
                     n_prompt_tokens=llm_resp.n_prompt_tokens)
            turn.add("llm_gen", llm_resp.decode_ms,
                     n_output_tokens=llm_resp.n_output_tokens)

            assistant_text = llm_resp.text

            # ── Check for final answer ─────────────────────────────────────────
            final_answer = extract_answer(assistant_text)
            if final_answer:
                record.add_turn(turn)
                record.final_answer = final_answer
                break

            # ── Extract SQL ────────────────────────────────────────────────────
            sql = extract_sql(assistant_text)
            used_fallback = False

            if sql is None:
                # LLM didn't produce SQL — if no fallback, end agent
                if self.fallback_sql is None:
                    record.sql_failure_count += 1
                    record.add_turn(turn)
                    record.final_answer = "(agent ended: no SQL generated)"
                    break
                sql = self.fallback_sql
                used_fallback = True

            # ── Execute SQL ────────────────────────────────────────────────────
            try:
                qr: QueryResult = self.db.execute(sql)
            except Exception as e:
                qr = QueryResult([], [], 0.0, 0.0, backend_name, str(e))

            if not qr.ok:
                record.sql_failure_count += 1
                if self.fallback_sql and not used_fallback:
                    # retry with gold SQL
                    qr = self.db.execute(self.fallback_sql)
                    used_fallback = True

            if qr.ok:
                record.sql_success_count += 1
            turn.add("sql_exec", qr.exec_ms, n_rows=qr.n_rows, n_cols=qr.n_cols)
            turn.add("fetch", qr.fetch_ms, n_rows=qr.n_rows)

            # ── Serialize result to markdown text ──────────────────────────────
            with StageTimer() as ser_timer:
                result_text = format_result_as_markdown(qr.columns, qr.rows)
            turn.add("serialize", ser_timer.elapsed_ms,
                     n_bytes=len(result_text.encode()))

            # ── Tokenize (CPU) ─────────────────────────────────────────────────
            with StageTimer() as tok_timer:
                encoding = self.tokenizer.encode(result_text)
            n_tokens = len(encoding.ids)
            turn.add("tokenize", tok_timer.elapsed_ms, n_tokens=n_tokens)

            record.add_turn(turn)

            # ── Build next message: append observation ─────────────────────────
            messages.append({"role": "assistant", "content": assistant_text})
            observation = f"Observation:\n{result_text}\n\nContinue your analysis."
            messages.append({"role": "user", "content": observation})

        else:
            # Exceeded max_turns
            record.final_answer = f"(agent ended: exceeded {self.max_turns} turns)"

        return record
