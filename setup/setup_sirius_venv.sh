#!/usr/bin/env bash
# Set up a dedicated venv with duckdb==1.4.3 for loading the Sirius extension.
#
# The Sirius extension is built for DuckDB v1.4.3, but the main env has 1.4.4.
# SIRIUS_ROOT defaults to /home/cc/sirius; override with the env var if needed:
#   SIRIUS_ROOT=/path/to/sirius bash setup/setup_sirius_venv.sh
#
# Usage: bash setup/setup_sirius_venv.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/../.venv/sirius"
mkdir -p "$SCRIPT_DIR/../.venv"
VENV_DIR="$(realpath "$VENV_DIR" 2>/dev/null || echo "$VENV_DIR")"

SIRIUS_ROOT="${SIRIUS_ROOT:-/home/cc/sirius}"
EXT_PATH="$SIRIUS_ROOT/build/release/extension/sirius/sirius.duckdb_extension"

echo "Creating Sirius Python venv at: $VENV_DIR"
python3 -m venv "$VENV_DIR"

echo "Installing duckdb==1.4.3 ..."
"$VENV_DIR/bin/pip" install --quiet duckdb==1.4.3

# Verify the extension loads
echo "Verifying Sirius extension loads..."
"$VENV_DIR/bin/python3" - "$EXT_PATH" <<'PYEOF'
import sys, duckdb
ext = sys.argv[1]
print(f"  duckdb version: {duckdb.__version__}")
con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
try:
    con.execute(f"load '{ext}'")
    print("  Sirius extension: loaded OK")
except Exception as e:
    print(f"  Sirius extension: FAILED — {e}")
    raise
PYEOF

echo ""
echo "Sirius venv ready at: $VENV_DIR"
echo "The SiriusGPUBackend will automatically use this Python interpreter."
