"""
Unified SQL backend — single class that handles both DuckDB CPU and Sirius GPU.

When use_sirius=True, queries are automatically wrapped in
CALL gpu_processing('...') and executed through the Sirius extension.

Timing model:
  - DuckDB CPU: exec_ms = execute(), fetch_ms = fetchall()
  - Sirius GPU: exec_ms = execute() + fetchall() (GPU work spans both), fetch_ms = 0
"""

import decimal
import datetime
import os
import re
from pathlib import Path
from typing import Optional

# Sirius extension may need libraries from its pixi environment.
# Pre-load libssl from pixi so the extension can find OPENSSL_3.2.0.
_SIRIUS_ROOT_PATH = Path(os.environ.get("SIRIUS_ROOT", "/home/ubuntu/sirius"))
_PIXI_LIB = _SIRIUS_ROOT_PATH / ".pixi/envs/default/lib"
if _PIXI_LIB.exists():
    import ctypes
    for _lib_name in ["libssl.so.3", "libcurl.so.4"]:
        _lib_path = _PIXI_LIB / _lib_name
        if _lib_path.exists():
            try:
                ctypes.CDLL(str(_lib_path))
            except OSError:
                pass

import duckdb

from core.timer import StageTimer
from core.backends.duckdb_cpu import QueryResult

# Sirius extension path — override with SIRIUS_EXT env var
_DEFAULT_SIRIUS_EXT = _SIRIUS_ROOT_PATH / "build/release/extension/sirius/sirius.duckdb_extension"


class SQLBackend:
    """
    Unified SQL backend supporting both DuckDB CPU and Sirius GPU execution.

    Usage:
        # CPU mode
        with SQLBackend("/path/to/tpch.duckdb") as db:
            result = db.execute("SELECT count(*) FROM lineitem")

        # GPU mode (Sirius)
        with SQLBackend("/path/to/tpch.duckdb", use_sirius=True) as db:
            result = db.execute("SELECT count(*) FROM lineitem")
    """

    def __init__(
        self,
        db_path: str,
        use_sirius: bool = False,
        sirius_ext_path: Optional[str] = None,
        gpu_cache_size: str = "4 GB",
        gpu_proc_size: str = "4 GB",
    ):
        self.db_path = db_path
        self.use_sirius = use_sirius
        self.sirius_ext_path = sirius_ext_path or str(_DEFAULT_SIRIUS_EXT)
        self.gpu_cache_size = gpu_cache_size
        self.gpu_proc_size = gpu_proc_size
        self._con: Optional[duckdb.DuckDBPyConnection] = None

    @property
    def name(self) -> str:
        return "sirius_gpu" if self.use_sirius else "duckdb_cpu"

    def connect(self):
        config = {}
        if self.use_sirius:
            config["allow_unsigned_extensions"] = "true"

        self._con = duckdb.connect(self.db_path, read_only=True, config=config)

        if self.use_sirius:
            ext = self.sirius_ext_path
            self._con.execute(f"LOAD '{ext}'")
            self._con.execute(
                f"CALL gpu_buffer_init('{self.gpu_cache_size}', '{self.gpu_proc_size}')"
            )

    def close(self):
        if self._con:
            self._con.close()
            self._con = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    def execute(self, sql: str) -> QueryResult:
        """Execute SQL and return QueryResult with timing."""
        if self._con is None:
            raise RuntimeError("Backend not connected. Use as context manager.")

        if self.use_sirius:
            return self._execute_sirius(sql)
        else:
            return self._execute_cpu(sql)

    def _execute_cpu(self, sql: str) -> QueryResult:
        """Standard DuckDB execution with separate exec/fetch timing."""
        with StageTimer() as exec_timer:
            cursor = self._con.execute(sql)
        with StageTimer() as fetch_timer:
            rows = cursor.fetchall()
            columns = (
                [desc[0] for desc in cursor.description]
                if cursor.description
                else []
            )
        return QueryResult(
            columns=columns,
            rows=rows,
            exec_ms=exec_timer.elapsed_ms,
            fetch_ms=fetch_timer.elapsed_ms,
            backend="duckdb_cpu",
            error=None,
        )

    def _execute_sirius(self, sql: str) -> QueryResult:
        """Sirius GPU execution — wraps query in gpu_processing()."""
        # Normalize: collapse whitespace, escape single quotes
        sql_norm = " ".join(sql.split()).replace("'", "''")
        wrapped = f"CALL gpu_processing('{sql_norm}')"

        with StageTimer() as timer:
            cursor = self._con.execute(wrapped)
            raw_rows = cursor.fetchall()
            columns = (
                [desc[0] for desc in cursor.description]
                if cursor.description
                else []
            )

        # Convert any non-JSON-serializable types
        rows = [
            tuple(_safe_value(v) for v in row) for row in raw_rows
        ]

        return QueryResult(
            columns=columns,
            rows=rows,
            exec_ms=timer.elapsed_ms,
            fetch_ms=0.0,  # GPU work spans execute + fetchall
            backend="sirius_gpu",
            error=None,
        )

    def warmup(self, sql: str):
        """Run a query to warm up caches (not timed)."""
        try:
            self.execute(sql)
        except Exception:
            pass


def _safe_value(v):
    """Convert DB values to Python-native types."""
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (datetime.date, datetime.datetime)):
        return str(v)
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v
