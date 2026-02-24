"""
Generate TPC-H datasets at multiple scale factors and save as .duckdb files.

Usage:
    python setup/generate_tpch.py            # generates SF=1, SF=5, SF=10
    python setup/generate_tpch.py --sf 1 5  # specific scale factors
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb

DATA_DIR = Path(__file__).parent.parent / "data"


def generate(sf: int, force: bool = False):
    out_path = DATA_DIR / f"tpch_sf{sf}.duckdb"
    if out_path.exists() and not force:
        print(f"  [SKIP] {out_path.name} already exists (use --force to regenerate)")
        return

    print(f"  Generating TPC-H SF={sf} → {out_path.name} ...", flush=True)
    t0 = time.time()

    DATA_DIR.mkdir(exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    con = duckdb.connect(str(out_path))
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute(f"CALL dbgen(sf={sf})")

    # Quick sanity check
    row_counts = {}
    for table in ["lineitem", "orders", "customer", "supplier", "part", "partsupp", "nation", "region"]:
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        row_counts[table] = n

    con.close()
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"    lineitem: {row_counts['lineitem']:,} rows")
    print(f"    orders:   {row_counts['orders']:,} rows")


def main():
    parser = argparse.ArgumentParser(description="Generate TPC-H datasets")
    parser.add_argument("--sf",    nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--force", action="store_true", help="Regenerate even if exists")
    args = parser.parse_args()

    print("Generating TPC-H datasets...")
    for sf in sorted(args.sf):
        generate(sf, force=args.force)
    print("\nDone. Files saved to:", DATA_DIR)


if __name__ == "__main__":
    main()
