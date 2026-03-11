"""
ReAct (Reason + Act) agent loop with per-stage timing and retry-until-correct.

The agent follows the standard ReAct pattern:
  Thought → Action (SQL) → Observation (result) → Thought → ... → Answer

Stage timing per turn:
  sql_exec    - DB executes the SQL (includes GPU compute for Sirius)
  fetch       - fetchall() copies results to Python
  serialize   - Format result table as markdown (CPU string work)
  tokenize    - Tokenize the formatted text (CPU tokenizer)
  llm_prefill - LLM processes context with SQL result
  llm_gen     - LLM generating SQL / reasoning / final answer (decode)

data_path = fetch + serialize + tokenize = what GPU colocation would eliminate.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

from tokenizers import Tokenizer

from core.backends.duckdb_cpu import QueryResult
from core.timer import RunRecord, StageTimer, TurnRecord


# ── LLM Protocol ─────────────────────────────────────────────────────────────

class LLMBackendProtocol(Protocol):
    """Any LLM backend must implement this interface."""
    def chat(self, messages: list[dict], prev_prompt_tokens: int = 0): ...


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are a data analyst assistant with access to a SQL database.
To answer the user's question, run SQL queries using this format:

Action: SQL
```sql
SELECT ...
```

After seeing the result (Observation), continue reasoning until done.
When you have a final answer, write:

Answer: <your final answer here>

Rules:
- One SQL block per Action, then stop and wait for the result.
- Use valid DuckDB SQL. No EXPLAIN, CREATE, INSERT, or DROP.
- Keep SQL concise; avoid SELECT * on large tables.
- Limit large scans to at most 1000 rows unless asked otherwise.
- If results are empty or unexpected, try a different approach.

{schema}
"""

RETRY_PROMPT = """\
Your previous answer was not quite right. Here is the question again:

{question}

Please try again carefully. Make sure your final Answer: line contains
the specific value asked for (a number, name, or brief result).
"""

# ── Result formatting ─────────────────────────────────────────────────────────

def format_result_as_markdown(columns: list[str], rows: list[tuple]) -> str:
    """Convert query result to a markdown table string."""
    if not rows:
        return "(empty result)"
    MAX_ROWS = 100  # cap to keep context within model limits
    lines = []
    lines.append("| " + " | ".join(str(c) for c in columns) + " |")
    lines.append("|" + "|".join(" --- " for _ in columns) + "|")
    for row in rows[:MAX_ROWS]:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    if len(rows) > MAX_ROWS:
        lines.append(f"*... {len(rows) - MAX_ROWS} more rows truncated*")
    return "\n".join(lines)


# ── SQL / answer extraction ───────────────────────────────────────────────────

_SQL_BLOCK = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_ANSWER    = re.compile(r"^Answer\s*:\s*(.+)", re.MULTILINE | re.DOTALL)


def extract_sql(text: str) -> Optional[str]:
    m = _SQL_BLOCK.search(text)
    return m.group(1).strip() if m else None


def extract_answer(text: str) -> Optional[str]:
    m = _ANSWER.search(text)
    return m.group(1).strip() if m else None


# ── Agent ─────────────────────────────────────────────────────────────────────

class ReactAgent:
    """
    ReAct agent with per-stage timing and retry-until-correct.

    Works with any LLM backend (LlamaBackend, VLLMBackend) and any SQL
    backend (SQLBackend, DuckDBCPUBackend, SiriusGPUBackend).
    """

    def __init__(
        self,
        llm,
        db,
        tokenizer: Tokenizer,
        max_turns: int = 10,
        fallback_sql: Optional[str] = None,
        max_retries: int = 2,
    ):
        self.llm = llm
        self.db = db
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.fallback_sql = fallback_sql
        self.max_retries = max_retries

    def run(
        self,
        question: str,
        schema_hint: str,
        task_name: str,
        backend_name: str,
        model_name: str,
        scale_factor: int,
        validation_key: Optional[str] = None,
    ) -> RunRecord:
        """Run the agent (with retries if validation_key is provided)."""
        record = RunRecord(
            task_name=task_name,
            backend=backend_name,
            model_name=model_name,
            scale_factor=scale_factor,
            mode="agentic",
        )

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(schema=schema_hint)

        for attempt in range(1 + self.max_retries):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question},
            ]

            answer = self._run_attempt(messages, record, backend_name)

            if validation_key is None:
                record.final_answer = answer or "(no answer)"
                record.answer_correct = True
                break

            correct = validation_key.lower() in (answer or "").lower()
            record.answer_correct = correct

            if correct:
                record.final_answer = answer or "(no answer)"
                break

            if attempt < self.max_retries:
                record.n_retries = attempt + 1
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": RETRY_PROMPT.format(question=question)},
                ]
            else:
                record.final_answer = answer or "(no answer after retries)"

        if not hasattr(record, "n_retries"):
            record.n_retries = 0
        if not hasattr(record, "answer_correct"):
            record.answer_correct = False

        return record

    def _run_attempt(
        self,
        messages: list[dict],
        record: RunRecord,
        backend_name: str,
    ) -> Optional[str]:
        """Run one attempt (max_turns turns). Returns final answer or None."""
        prev_prompt_tokens = 0
        for turn_idx in range(self.max_turns):
            turn = TurnRecord(turn_idx=len(record.turns))

            # ── LLM generates Thought + Action ────────────────────────────
            llm_resp = self.llm.chat(messages, prev_prompt_tokens=prev_prompt_tokens)
            turn.add("llm_prefill", llm_resp.prefill_ms,
                     n_prompt_tokens=llm_resp.n_prompt_tokens)
            turn.add("llm_prefill_incr", llm_resp.prefill_incr_ms,
                     n_new_prompt_tokens=llm_resp.n_new_prompt_tokens)
            turn.add("llm_gen", llm_resp.decode_ms,
                     n_output_tokens=llm_resp.n_output_tokens)
            prev_prompt_tokens = llm_resp.n_prompt_tokens

            assistant_text = llm_resp.text

            # ── Check for final Answer ─────────────────────────────────────
            final_answer = extract_answer(assistant_text)
            if final_answer:
                record.add_turn(turn)
                return final_answer

            # ── Extract SQL ────────────────────────────────────────────────
            sql = extract_sql(assistant_text)
            used_fallback = False

            if sql is None:
                if self.fallback_sql is None:
                    record.sql_failure_count += 1
                    record.add_turn(turn)
                    return None
                sql = self.fallback_sql
                used_fallback = True

            # ── Execute SQL ────────────────────────────────────────────────
            try:
                qr: QueryResult = self.db.execute(sql)
            except Exception as e:
                qr = QueryResult([], [], 0.0, 0.0, backend_name, str(e))

            if not qr.ok:
                record.sql_failure_count += 1
                if self.fallback_sql and not used_fallback:
                    try:
                        qr = self.db.execute(self.fallback_sql)
                    except Exception:
                        pass

            if qr.ok:
                record.sql_success_count += 1
            turn.add("sql_exec", qr.exec_ms, n_rows=qr.n_rows, n_cols=qr.n_cols)
            turn.add("fetch", qr.fetch_ms, n_rows=qr.n_rows)

            # ── Serialize result ───────────────────────────────────────────
            with StageTimer() as ser_timer:
                result_text = format_result_as_markdown(qr.columns, qr.rows)
            turn.add("serialize", ser_timer.elapsed_ms,
                     n_bytes=len(result_text.encode()))

            # ── Tokenize (CPU) ─────────────────────────────────────────────
            with StageTimer() as tok_timer:
                encoding = self.tokenizer.encode(result_text)
            turn.add("tokenize", tok_timer.elapsed_ms,
                     n_tokens=len(encoding.ids))

            record.add_turn(turn)

            # ── Append observation to conversation ─────────────────────────
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append({
                "role": "user",
                "content": f"Observation:\n{result_text}\n\nContinue your analysis.",
            })

        return None  # max_turns exceeded without an Answer
