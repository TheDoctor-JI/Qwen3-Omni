#!/bin/bash
# Single 72 GB GPU by default; select two with CUDA_VISIBLE_DEVICES=1,2.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use PCI ordering so CUDA and nvidia-smi indices agree on this host.
# This vLLM version requires numeric CUDA_VISIBLE_DEVICES (not UUIDs).
# On this host, nvidia-smi GPUs 1 and 2 are RTX PRO 5000 72GB cards.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-1}"
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)?$ ]]; then
    echo "ERROR: Set CUDA_VISIBLE_DEVICES to one or two GPU indices (e.g. 1 or 1,2)." >&2
    exit 1
fi
IFS=, read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#GPU_IDS[@]} -eq 2 && "${GPU_IDS[0]}" == "${GPU_IDS[1]}" ]]; then
    echo "ERROR: Select two distinct GPUs." >&2
    exit 1
fi

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-qwen3omni-vllm015}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES
for GPU_ID in "${GPU_IDS[@]}"; do
    nvidia-smi --id="$GPU_ID" --query-gpu=index,name,memory.total,uuid --format=csv,noheader
done

# Verify identities before loading weights; fail rather than use a wrong card
# on hosts whose NVML indices differ from PCI order.
python - <<'PY'
import os
import subprocess
import torch

selected = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
if torch.cuda.device_count() != len(selected):
    raise SystemExit('ERROR: CUDA cannot see all selected GPUs.')
for logical_id, physical_id in enumerate(selected):
    expected = subprocess.check_output([
        'nvidia-smi', '--id=' + physical_id,
        '--query-gpu=uuid', '--format=csv,noheader',
    ], text=True).strip().removeprefix('GPU-').lower()
    props = torch.cuda.get_device_properties(logical_id)
    actual = str(props.uuid).removeprefix('GPU-').lower()
    if actual != expected:
        raise SystemExit(f'ERROR: CUDA GPU {logical_id} does not match nvidia-smi GPU {physical_id}.')
    print(f'Verified CUDA GPU {logical_id}: {props.name}, {props.total_memory / 1024**3:.2f} GiB')
PY

echo "Starting Qwen3-Omni with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, tensor parallel size ${#GPU_IDS[@]}, memory utilization ${GPU_MEMORY_UTILIZATION:-0.9}"
exec python socketio_server.py \
    --checkpoint-path "${CHECKPOINT_PATH:-./Qwen3-Omni-30B-A3B-Thinking}" \
    --config "${CONFIG_PATH:-../AudioLLMInterface/MultiModalLLM/mm_llm_config.yaml}" \
    --host "${HOST:-0.0.0.0}" --port "${PORT:-8902}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
    --tensor-parallel-size "${#GPU_IDS[@]}" \
    "$@"
