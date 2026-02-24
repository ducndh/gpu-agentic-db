"""
DuckDB CPU backend — uses the standard duckdb Python package in-process.
No subprocess overhead; timing is as accurate as possible.
"""

import duckdb
from core.timer import StageTimer


class DuckDBCPUBackend:
    """
    Wraps a DuckDB connection for in-process CPU query execution.

    Usage:
        backend = DuckDBCPUBackend("/path/to/tpch.duckdb")
        result = backend.execute("SELECT * FROM lineitem LIMIT 10")
        print(result.columns, result.rows, result.exec_ms, result.fetch_ms)
    """

    name = "duckdb_cpu"

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._con: duckdb.DuckDBPyConnection | None = None

    def connect(self):
        self._con = duckdb.connect(self.db_path, read_only=True)

    def close(self):
        if self._con:
            self._con.close()
            self._con = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    def execute(self, sql: str) -> "QueryResult":
        """Execute SQL and return a QueryResult with timing."""
        if self._con is None:
            raise RuntimeError("Backend not connected. Use as context manager.")

        with StageTimer() as exec_timer:
            cursor = self._con.execute(sql)
        exec_ms = exec_timer.elapsed_ms

        with StageTimer() as fetch_timer:
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
        fetch_ms = fetch_timer.elapsed_ms

        return QueryResult(
            columns=columns,
            rows=rows,
            exec_ms=exec_ms,
            fetch_ms=fetch_ms,
            backend=self.name,
            error=None,
        )

    def warmup(self, sql: str):
        """Run a query once to warm up the connection/plan cache (not timed)."""
        try:
            self._con.execute(sql).fetchall()
        except Exception:
            pass


class QueryResult:
    """Container for query results + per-stage timing."""

    def __init__(
        self,
        columns: list[str],
        rows: list[tuple],
        exec_ms: float,
        fetch_ms: float,
        backend: str,
        error: str | None,
    ):
        self.columns = columns
        self.rows = rows
        self.exec_ms = exec_ms
        self.fetch_ms = fetch_ms
        self.backend = backend
        self.error = error
        self.n_rows = len(rows)
        self.n_cols = len(columns)

    @property
    def ok(self) -> bool:
        return self.error is None

    def __repr__(self):
        return (
            f"QueryResult(backend={self.backend}, rows={self.n_rows}, "
            f"exec={self.exec_ms:.1f}ms, fetch={self.fetch_ms:.2f}ms, "
            f"ok={self.ok})"
        )
