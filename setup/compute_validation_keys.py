"""
Compute validation keys (expected answer values) for each TPC-H task at each SF.

Runs gold SQL against each .duckdb file and extracts a distinctive substring
from the result that should appear in the agent's final answer.

Usage:
    python setup/compute_validation_keys.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb

from tasks.tpch import TASKS

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def extract_key(columns, rows) -> str | None:
    """
    Extract a validation key from a result set.
    For 1-row results: return the first numeric value as a string.
    For multi-row: return the first value of the first row.
    """
    if not rows:
        return None
    first_row = rows[0]
    # Find the first non-null value and convert to a recognizable string
    for v in first_row:
        if v is not None:
            s = str(v)
            # For numbers: use a short prefix (first 6 chars) to avoid float precision issues
            if s.replace(".", "").replace("-", "").isdigit():
                return s[:8]
            return s[:20]
    return None


def main():
    keys: dict[str, dict[int, str]] = {}

    for sf in [1, 5, 10]:
        db_file = DATA_DIR / f"tpch_sf{sf}.duckdb"
        if not db_file.exists():
            print(f"[SKIP] tpch_sf{sf}.duckdb not found")
            continue

        con = duckdb.connect(str(db_file), read_only=True)
        print(f"\nSF={sf}:")

        for task in TASKS:
            try:
                cur = con.execute(task.gold_sql)
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                key = extract_key(cols, rows)
                if task.name not in keys:
                    keys[task.name] = {}
                keys[task.name][sf] = key
                print(f"  {task.name:<30} rows={len(rows)}  key={key!r}")
            except Exception as e:
                print(f"  {task.name:<30} ERROR: {e}")

        con.close()

    # Print as Python dict for copy-paste into tasks.py
    print("\n\n# Validation keys by task and SF:")
    print("VALIDATION_KEYS =", json.dumps(keys, indent=2))

    output = PROJECT_ROOT / "results" / "validation_keys.json"
    with open(output, "w") as f:
        json.dump(keys, f, indent=2, default=str)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
