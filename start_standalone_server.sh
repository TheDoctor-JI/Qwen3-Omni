#!/usr/bin/env bash
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate qwen3omni-vllm015
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python -m qwen_service start "$@"
