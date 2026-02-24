"""
Sirius GPU backend — runs a long-lived worker subprocess that holds
a DuckDB+Sirius connection open.  The main process sends queries via stdin
as JSON and reads results+timing from stdout as JSON.

This avoids per-query process startup overhead while still being fully
isolated from the main Python environment (which has duckdb 1.4.4, whereas
the Sirius extension requires 1.4.3).

Worker protocol (newline-delimited JSON):
  Request:  {"id": "1", "action": "query",    "sql": "SELECT ..."}
            {"id": "2", "action": "shutdown"}
  Response: {"id": "1", "ok": true,  "columns": [...], "rows": [...],
             "exec_ms": 12.3, "fetch_ms": 0.4}
            {"id": "1", "ok": false, "error": "..."}
"""

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

from core.backends.duckdb_cpu import QueryResult
from core.timer import StageTimer

# Path to Sirius build
SIRIUS_ROOT = Path("/home/cc/sirius")
SIRIUS_EXT = SIRIUS_ROOT / "build/release/extension/sirius/sirius.duckdb_extension"

# Default GPU buffer sizes (leave headroom for LLM)
GPU_CACHE_SIZE = "4 GB"
GPU_PROC_SIZE = "2 GB"


def _worker_script(db_path: str, ext_path: str, cache: str, proc: str) -> str:
    """Generate the Python worker script source code."""
    return textwrap.dedent(f"""\
        import sys, json, time, traceback
        import duckdb

        db_path  = {db_path!r}
        ext_path = {ext_path!r}
        cache    = {cache!r}
        proc     = {proc!r}

        # Connect with unsigned extensions allowed
        con = duckdb.connect(db_path, config={{"allow_unsigned_extensions": "true"}})
        con.execute(f"load '{{ext_path}}'")
        con.execute(f"call gpu_buffer_init('{{cache}}', '{{proc}}')")

        # Signal ready
        print(json.dumps({{"ready": True}}), flush=True)

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            rid = req.get("id", "?")

            if req.get("action") == "shutdown":
                break

            sql = req.get("sql", "")
            try:
                t0 = time.perf_counter()
                cursor = con.execute(f'call gpu_processing("{{sql}}")')
                t1 = time.perf_counter()
                rows = cursor.fetchall()
                t2 = time.perf_counter()
                columns = [d[0] for d in cursor.description] if cursor.description else []
                # gpu_processing returns a result set with one column 'result'
                # which contains the actual query output as a table — unwrap it
                # Actually sirius returns rows directly; columns come from description
                resp = {{
                    "id": rid, "ok": True,
                    "columns": columns,
                    "rows": [list(r) for r in rows],
                    "exec_ms": (t1 - t0) * 1000,
                    "fetch_ms": (t2 - t1) * 1000,
                }}
            except Exception as e:
                resp = {{"id": rid, "ok": False, "error": str(e)}}

            print(json.dumps(resp), flush=True)

        con.close()
    """)


class SiriusGPUBackend:
    """
    Wraps the Sirius GPU query engine via a long-running worker subprocess.

    The worker keeps the GPU buffer initialized between queries, so you pay
    the gpu_buffer_init cost only once at startup.

    Usage:
        backend = SiriusGPUBackend("/path/to/tpch.duckdb")
        with backend:
            result = backend.execute("SELECT count(*) FROM lineitem")
    """

    name = "sirius_gpu"

    def __init__(
        self,
        db_path: str,
        gpu_cache: str = GPU_CACHE_SIZE,
        gpu_proc: str = GPU_PROC_SIZE,
        startup_timeout: float = 60.0,
    ):
        self.db_path = db_path
        self.gpu_cache = gpu_cache
        self.gpu_proc = gpu_proc
        self.startup_timeout = startup_timeout
        self._proc: subprocess.Popen | None = None
        self._req_id = 0
        self._lock = threading.Lock()
        self._ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self):
        """Start the worker subprocess and wait for it to signal ready."""
        script = _worker_script(
            db_path=self.db_path,
            ext_path=str(SIRIUS_EXT),
            cache=self.gpu_cache,
            proc=self.gpu_proc,
        )

        # Find duckdb 1.4.3 Python — prefer venv if set up, else try system
        python_bin = self._find_python()

        self._proc = subprocess.Popen(
            [python_bin, "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )

        # Wait for {"ready": true}
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                stderr = self._proc.stderr.read()
                raise RuntimeError(
                    f"Sirius worker exited during startup. stderr:\n{stderr}"
                )
            line = self._proc.stdout.readline().strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if msg.get("ready"):
                    self._ready = True
                    return
            except json.JSONDecodeError:
                continue

        raise TimeoutError(
            f"Sirius worker did not signal ready within {self.startup_timeout}s"
        )

    def close(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._send({"action": "shutdown"})
                self._proc.wait(timeout=5.0)
            except Exception:
                self._proc.kill()
        self._proc = None
        self._ready = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def execute(self, sql: str) -> QueryResult:
        """Send a query to the Sirius worker and return QueryResult with timing."""
        if not self._ready or self._proc is None:
            raise RuntimeError("Backend not connected.")

        with self._lock:
            self._req_id += 1
            rid = str(self._req_id)
            request = json.dumps({"id": rid, "action": "query", "sql": sql})

            with StageTimer() as total_timer:
                self._proc.stdin.write(request + "\n")
                self._proc.stdin.flush()

                # Read response (blocking)
                while True:
                    if self._proc.poll() is not None:
                        stderr = self._proc.stderr.read()
                        raise RuntimeError(f"Sirius worker died. stderr:\n{stderr}")
                    line = self._proc.stdout.readline().strip()
                    if not line:
                        continue
                    try:
                        resp = json.loads(line)
                        if resp.get("id") == rid:
                            break
                    except json.JSONDecodeError:
                        continue

        if not resp.get("ok"):
            return QueryResult(
                columns=[],
                rows=[],
                exec_ms=0.0,
                fetch_ms=0.0,
                backend=self.name,
                error=resp.get("error", "Unknown error"),
            )

        return QueryResult(
            columns=resp["columns"],
            rows=[tuple(r) for r in resp["rows"]],
            exec_ms=resp["exec_ms"],
            fetch_ms=resp["fetch_ms"],
            backend=self.name,
            error=None,
        )

    def warmup(self, sql: str):
        """Run query once to warm up Sirius caches (not timed)."""
        try:
            self.execute(sql)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send(self, msg: dict):
        if self._proc and self._proc.stdin:
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()

    @staticmethod
    def _find_python() -> str:
        """Find the Python interpreter that has duckdb 1.4.3 for Sirius."""
        # Priority: project venv > conda libcudf-env > current interpreter
        candidates = [
            "/home/cc/gpu-agentic-db/.venv/sirius/bin/python",
            "/home/cc/miniconda3/envs/libcudf-env/bin/python3",
            sys.executable,
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return sys.executable
