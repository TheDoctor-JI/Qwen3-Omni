#!/bin/bash
# ============================================================================
# Launch Qwen3-4B Socket.IO Server
# ============================================================================
# Usage:
#   bash start_socketio_server_qwen3_4b.sh
#
# This script loads the Qwen3-4B checkpoint (text-only, thinking-capable)
# via vLLM and exposes the same Socket.IO interface as the Omni server.
# All multimodal paths are automatically bypassed when the server detects a
# non-Omni checkpoint.
# ============================================================================

set -euo pipefail

# For releasing ports — kill any existing processes on port 8902
lsof -t -i:8902 | xargs -r kill -9 2>/dev/null || true

# Also kill any lingering vLLM / socketio_server processes
pkill -9 -f "socketio_server.py" 2>/dev/null || true

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3omni

# Launch the Socket.IO server
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_PATH="../AudioLLMInterface/MultiModalLLM/mm_llm_config.yaml"

# Qwen3-4B checkpoint (from HuggingFace cache)
QWEN3_4B_PATH="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/82d62bb073771e7a1ea59435f548908540217d1f"

echo "============================================================"
echo " Starting Qwen3-4B Socket.IO Server"
echo " Checkpoint: $QWEN3_4B_PATH"
echo " Listen:     0.0.0.0:8902"
echo " GPU:        CUDA:1 (single GPU)"
echo " Profile:    text-only (thinking mode supported)"
echo "============================================================"

CUDA_VISIBLE_DEVICES=1 python socketio_server.py \
    --checkpoint-path "$QWEN3_4B_PATH" \
    --config "$CONFIG_PATH" \
    --host 0.0.0.0 --port 8902 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1
