#!/usr/bin/env bash
# Setup script for running the benchmark on L40S with vLLM
# Run this on the L40S machine after cloning the repo.
#
# L40S: Ada Lovelace, 48GB GDDR6, compute capability 8.9
# Supports: Flash Attention 3, FP8, BF16, AWQ, GPTQ

set -e
cd "$(dirname "$0")/.."

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── Python environment ────────────────────────────────────────────────────────
echo "=== Creating venv ==="
python3 -m venv .venv/l40s
source .venv/l40s/bin/activate

pip install --upgrade pip

# vLLM — installs torch, flash-attn, xformers automatically
# Use the pre-built wheel for CUDA 12.x (L40S requires CUDA 12+)
pip install vllm

# Other dependencies
pip install duckdb tokenizers tabulate huggingface_hub

echo ""
echo "=== vLLM version ==="
python -c "import vllm; print(vllm.__version__)"

echo ""
echo "=== Flash Attention ==="
python -c "
from vllm.attention.backends.flash_attn import FlashAttentionBackend
print('Flash Attention available:', True)
" 2>/dev/null || echo "Flash Attention: check vLLM logs on first run"

# ── TPC-H data ────────────────────────────────────────────────────────────────
echo ""
echo "=== Generating TPC-H data (SF 1, 5, 10) ==="
mkdir -p data results
python setup/generate_tpch.py

# ── Validation keys ───────────────────────────────────────────────────────────
echo ""
echo "=== Computing validation keys ==="
python setup/compute_validation_keys.py

# ── Model download ────────────────────────────────────────────────────────────
echo ""
echo "=== Downloading models from HuggingFace ==="
echo "  Models will be cached in ~/.cache/huggingface/"
echo "  Estimated sizes:"
echo "    Qwen2.5-7B-Instruct  (BF16): ~15GB"
echo "    Qwen2.5-14B-Instruct (BF16): ~29GB"
echo "    Qwen2.5-32B-Instruct-AWQ   : ~19GB"
echo ""

python - <<'EOF'
from huggingface_hub import snapshot_download
import sys

models = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct-AWQ",
]

for model_id in models:
    print(f"\nDownloading {model_id}...")
    try:
        path = snapshot_download(repo_id=model_id, local_files_only=False)
        print(f"  -> {path}")
    except Exception as e:
        print(f"  [WARN] Failed: {e}")
        print(f"  Run manually: huggingface-cli download {model_id}")
EOF

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run experiment C (full sweep):"
echo "  source .venv/l40s/bin/activate"
echo "  python experiments/exp_c_vllm_l40s.py"
echo ""
echo "Quick smoke test (1 model, 1 task, 1 SF):"
echo "  python experiments/exp_c_vllm_l40s.py --models qwen2.5-7b-bf16 --tasks q6 --sf 1 --backends duckdb_cpu"
echo ""
echo "Compare with vs without prefix caching:"
echo "  python experiments/exp_c_vllm_l40s.py --output results/exp_c_with_cache.json"
echo "  python experiments/exp_c_vllm_l40s.py --output results/exp_c_no_cache.json --no-prefix-cache"
echo ""
echo "Run report across all experiments:"
echo "  python analysis/report.py results/exp_a_fixed_sql.json results/exp_b_agentic.json results/exp_c_vllm_l40s.json"
