#!/bin/bash


# Production always binds 8899. An occupied port must fail without killing its owner.
# Other model servers are independent of this launch.


# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3omni-vllm015

# Launch the Socket.IO server
cd "$(dirname "${BASH_SOURCE[0]}")"
CONFIG_PATH="../AudioLLMInterface/MultiModalLLM/mm_llm_config.yaml"
# The launcher writes model.checkpoint (thinking or instruct) into the YAML.
exec python ../runtime_launch.py --config "$CONFIG_PATH" --section model --production-qwen --append-port --append-checkpoint -- python socketio_server.py \
    --config "$CONFIG_PATH" \
    --host 0.0.0.0
