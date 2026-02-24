"""
Per-stage nanosecond timer for the agentic pipeline.

Stages tracked:
  sql_exec    - Query execution inside the database (CPU or GPU)
  fetch       - fetchall(): copying results from DB into Python objects
  serialize   - Formatting rows into markdown text for the LLM context
  tokenize    - CPU tokenizer: text → token IDs
  llm_gen     - LLM generating SQL query or final answer (decode phase)
  llm_prefill - LLM processing new context (prefill phase, estimated from token count)

The "data path" (fetch + serialize + tokenize) is what GPU colocation would eliminate.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

DATA_PATH_STAGES = ("fetch", "serialize", "tokenize")
ALL_STAGES = ("sql_exec", "fetch", "serialize", "tokenize", "llm_prefill", "llm_prefill_incr", "llm_gen")


@dataclass
class StageRecord:
    stage: str
    duration_ms: float
    metadata: dict = field(default_factory=dict)


@dataclass
class TurnRecord:
    """Timing record for a single agent turn (one LLM call + optional SQL call)."""
    turn_idx: int
    records: list[StageRecord] = field(default_factory=list)

    def add(self, stage: str, duration_ms: float, **metadata):
        self.records.append(StageRecord(stage, duration_ms, metadata))

    def get(self, stage: str) -> Optional[float]:
        """Sum of all durations for a given stage in this turn."""
        total = sum(r.duration_ms for r in self.records if r.stage == stage)
        return total if total > 0 else None

    def total_ms(self) -> float:
        return sum(r.duration_ms for r in self.records)

    def data_path_ms(self) -> float:
        return sum(
            r.duration_ms for r in self.records if r.stage in DATA_PATH_STAGES
        )

    def data_path_pct(self) -> float:
        total = self.total_ms()
        return (self.data_path_ms() / total * 100) if total > 0 else 0.0


@dataclass
class RunRecord:
    """Timing record for a complete agent run (multiple turns)."""
    task_name: str
    backend: str          # "duckdb_cpu" | "sirius_gpu"
    model_name: str
    scale_factor: int
    mode: str             # "agentic" | "fixed_sql"
    turns: list[TurnRecord] = field(default_factory=list)
    sql_success_count: int = 0
    sql_failure_count: int = 0
    final_answer: str = ""

    def add_turn(self, turn: TurnRecord):
        self.turns.append(turn)

    def total_ms(self) -> float:
        return sum(t.total_ms() for t in self.turns)

    def total_data_path_ms(self) -> float:
        return sum(t.data_path_ms() for t in self.turns)

    def data_path_pct(self) -> float:
        total = self.total_ms()
        return (self.total_data_path_ms() / total * 100) if total > 0 else 0.0

    def colocation_ceiling_ms(self) -> float:
        """Theoretical minimum latency if data path cost → 0."""
        return self.total_ms() - self.total_data_path_ms()

    def colocation_speedup(self) -> float:
        ceiling = self.colocation_ceiling_ms()
        return (self.total_ms() / ceiling) if ceiling > 0 else 1.0

    def total_incremental_llm_ms(self) -> float:
        """LLM cost using KV-cache-aware incremental prefill + decode across all turns."""
        return sum(
            r.duration_ms for t in self.turns for r in t.records
            if r.stage in ("llm_prefill_incr", "llm_gen")
        )

    def incr_colocation_ceiling_ms(self) -> float:
        """Per-turn ceiling: total - data_path, with incremental LLM as denominator.
        Represents the fraction of per-turn work colocation can eliminate."""
        return self.total_incremental_llm_ms() + (self.total_ms() - self.total_incremental_llm_ms() - self.total_data_path_ms())

    def incr_data_path_pct(self) -> float:
        """data_path as % of (data_path + incremental LLM) — the true per-turn ratio."""
        denom = self.total_data_path_ms() + self.total_incremental_llm_ms()
        return (self.total_data_path_ms() / denom * 100) if denom > 0 else 0.0

    def stage_breakdown(self) -> dict[str, float]:
        """Total ms per stage across all turns."""
        breakdown: dict[str, float] = {s: 0.0 for s in ALL_STAGES}
        for turn in self.turns:
            for r in turn.records:
                if r.stage in breakdown:
                    breakdown[r.stage] += r.duration_ms
        return breakdown

    def to_dict(self) -> dict:
        breakdown = self.stage_breakdown()
        incr_llm = self.total_incremental_llm_ms()
        d = {
            "task": self.task_name,
            "backend": self.backend,
            "model": self.model_name,
            "scale_factor": self.scale_factor,
            "mode": self.mode,
            "n_turns": len(self.turns),
            "sql_success": self.sql_success_count,
            "sql_failure": self.sql_failure_count,
            "total_ms": round(self.total_ms(), 2),
            "data_path_ms": round(self.total_data_path_ms(), 2),
            "data_path_pct": round(self.data_path_pct(), 1),
            "colocation_ceiling_ms": round(self.colocation_ceiling_ms(), 2),
            "colocation_speedup": round(self.colocation_speedup(), 3),
            "incr_llm_ms": round(incr_llm, 2),
            "incr_data_path_pct": round(self.incr_data_path_pct(), 1),
            "stage_breakdown_ms": {k: round(v, 2) for k, v in breakdown.items()},
            "final_answer": self.final_answer,
        }
        # Optional fields set by ReactAgent
        if hasattr(self, "answer_correct"):
            d["answer_correct"] = self.answer_correct  # type: ignore[attr-defined]
        if hasattr(self, "n_retries"):
            d["n_retries"] = self.n_retries  # type: ignore[attr-defined]
        return d


class StageTimer:
    """Context manager for timing a single stage."""

    def __init__(self):
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
