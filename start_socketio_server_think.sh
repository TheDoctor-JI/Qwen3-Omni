#!/bin/bash


# Production always binds 8899. An occupied port must fail without killing its owner.
# Other model servers are independent of this launch.


# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3omni-vllm015

# Launch the Socket.IO server
cd "$(dirname "${BASH_SOURCE[0]}")"
CONFIG_PATH="../AudioLLMInterface/MultiModalLLM/mm_llm_config.yaml"
# If you want to switch checkpoints, edit the line below rather than
# placing a commented argument inside the continued command.
# --checkpoint-path ./Qwen3-Omni-30B-A3B-Instruct \
# --checkpoint-path ./Qwen3-Omni-30B-A3B-Thinking \

exec python ../runtime_launch.py --config "$CONFIG_PATH" --section model --production-qwen --append-port -- python socketio_server.py \
    --checkpoint-path ./Qwen3-Omni-30B-A3B-Thinking \
    --config "$CONFIG_PATH" \
    --host 0.0.0.0
