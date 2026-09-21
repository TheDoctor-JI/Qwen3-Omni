# Qwen3-Omni — Setup Guide

## Requirements
- NVIDIA **B200** (or other Blackwell / 50-series data-centre) GPU(s) with **≥79 GB** total VRAM (BF16)
- CUDA **12.8** driver (`nvidia-smi` should report ≥ 12.8)
- Miniconda / Anaconda

---

## 1. Create conda environment

```bash
conda create -n qwen3omni python=3.11 -y
conda activate qwen3omni
```

## 2. Install PyTorch (CUDA 12.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 3. Install vLLM

```bash
pip install vllm==0.13.0
```

## 4. Install Transformers and utilities

```bash
pip install transformers==4.57.3 accelerate
pip install "huggingface_hub>=0.34.0,<1.0"
pip install qwen-omni-utils -U
```

## 5. Install Gradio and audio dependencies (web_demo.py)

```bash
pip install gradio==5.44.1 gradio_client==1.12.1 soundfile==0.13.1
```

## 6. Install Socket.IO dependencies (socketio_server.py)

```bash
pip install "python-socketio[asyncio_client]" aiohttp
```

## 7. Install FlashAttention 2 (optional but recommended)

```bash
pip install -U flash-attn --no-build-isolation
```

## 8. Install ffmpeg

```bash
conda install -c conda-forge ffmpeg -y
```

## 9. Download the model

```bash
# Hugging Face
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir ./Qwen3-Omni-30B-A3B-Instruct

# Or ModelScope (Mainland China)
pip install -U modelscope
modelscope download --model Qwen/Qwen3-Omni-30B-A3B-Instruct --local_dir ./Qwen3-Omni-30B-A3B-Instruct
```

---

## Launch

### Option A — Gradio web demo (`web_demo.py`)

```bash
python web_demo.py -c ./Qwen3-Omni-30B-A3B-Instruct \
    --server-port 8901 --server-name 0.0.0.0
```

Open `http://<server-ip>:8901` in your browser.

### Option B — Socket.IO streaming server (`socketio_server.py`)

```bash
python socketio_server.py \
    --checkpoint-path ./Qwen3-Omni-30B-A3B-Instruct \
    --host 0.0.0.0 --port 8902
```

Open `http://<server-ip>:8902` in your browser.  
The built-in GUI supports text, file upload (audio / image / video),
live microphone recording with waveform visualization, and real-time
token-by-token streaming output.

### Socket.IO on one or two 72 GB GPUs

The alternative launcher defaults to GPU **1**, an RTX PRO 5000 72GB on
this host, with tensor parallel size **1** and GPU memory utilization **0.9**.
These settings override the shared YAML's GPU settings.

```bash
# Try one 72 GB GPU
./start_socketio_server_72gb.sh

# Use both 72 GB GPUs if one runs out of memory
CUDA_VISIBLE_DEVICES=1,2 ./start_socketio_server_72gb.sh

# Thinking checkpoint on one GPU
CHECKPOINT_PATH=./Qwen3-Omni-30B-A3B-Thinking ./start_socketio_server_72gb.sh
```

Check GPU indices with `nvidia-smi` on other hosts. The launcher interprets numeric
`CUDA_VISIBLE_DEVICES` selections as **nvidia-smi indices**, sets
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, and retains numeric IDs for vLLM compatibility.
It verifies CUDA device UUIDs against the selected nvidia-smi cards before loading
weights and stops if they disagree. It prints each selected card's name, memory,
and UUID at startup. Inside the process,
CUDA renumbers the selected cards starting at GPU 0.
Tensor parallel size follows
the number of selected GPUs. Single-card fit is not guaranteed: the server
retains its 32K context limit and shared YAML concurrency/multimodal settings,
which also consume memory. The two-GPU option is an explicit retry, not an
automatic fallback. This launcher does not stop existing servers; stop the old
server before replacing it, or set `PORT=8903` to use another port (with enough
free GPU memory).

Optional environment overrides: `GPU_MEMORY_UTILIZATION`, `CHECKPOINT_PATH`,
`CONFIG_PATH`, `HOST`, `PORT`, `CONDA_SH`, and `CONDA_ENV`. Relative paths resolve
from the Qwen3-Omni directory. Additional server CLI arguments can be appended.

### Option C — Transformers backend (streaming + audio output)

```bash
python web_demo.py -c ./Qwen3-Omni-30B-A3B-Instruct \
    --use-transformers --flash-attn2 \
    --server-port 8901 --server-name 0.0.0.0

# Add --generate-audio to also produce speech output
```

---

## Features

| Feature | web_demo (vLLM) | socketio_server (vLLM) | web_demo (Transformers) |
|---|---|---|---|
| Token streaming | ✅ Gradio SSE | ✅ Socket.IO WebSocket | ✅ `TextIteratorStreamer` |
| Thinking mode toggle | ✅ | ✅ | ✅ |
| Live mic recording | ❌ | ✅ (MediaRecorder + waveform) | ❌ |
| Multimodal upload (audio/image/video) | ✅ | ✅ (base64 over WS) | ✅ |
| Audio (speech) output | ❌ | ❌ | ✅ (`--generate-audio`) |
| Live metrics (TTFT / TPS) | terminal only | ✅ in-browser | terminal only |
| Speed | Fast | Fast | Slower |

- **TTFT** and total generation time are printed to the terminal on each request.
- **Thinking mode** can be toggled on/off in the UI sidebar at runtime.
