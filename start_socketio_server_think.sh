#!/bin/bash

# For releasing ports - kill any existing processes on port 8902
lsof -t -i:8902 | xargs -r kill -9 2>/dev/null || true

# Also kill any lingering Qwen3-Omni / vLLM processes related to this server
pkill -9 -f "socketio_server.py" 2>/dev/null || true


# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3omni

# Launch the Socket.IO server
cd "$(dirname "${BASH_SOURCE[0]}")"
CONFIG_PATH="../AudioLLMInterface/MultiModalLLM/mm_llm_config.yaml"
python socketio_server.py \
    --checkpoint-path ./Qwen3-Omni-30B-A3B-Thinking \
    --config "$CONFIG_PATH" \
    --host 0.0.0.0 --port 8902
