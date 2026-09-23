# Qwen3-Omni — Setup Guide

## Requirements
- A **72GB Blackwell** passed the current vLLM server smoke test (BF16, 32K context, max_num_seqs=2). Larger multimodal workloads or the full Transformers model can require more memory.
- CUDA **12.8** driver (`nvidia-smi` should report ≥ 12.8)
- Miniconda / Anaconda

---

## Server environment: vLLM 0.15.0

All Socket.IO launch scripts in this directory now default to
`qwen3omni-vllm015`. The old `qwen3omni` environment can remain installed.
The 72GB launcher, where present, still supports an explicit `CONDA_ENV` override.

From this directory:

```bash
conda create -n qwen3omni-vllm015 python=3.11 pip -y
conda activate qwen3omni-vllm015
python -m pip install -r requirements-server-vllm015.txt
python -m pip check
```

This installs vLLM 0.15.0, its required PyTorch 2.9.1 stack, Transformers 4.57.5,
and the Socket.IO/audio dependencies. vLLM's native Qwen audio encoder avoids
our old external FlashAttention binary's Blackwell incompatibility. Do not
install external `flash-attn` for this vLLM server. Keep FFmpeg on PATH:

```bash
conda install -c conda-forge ffmpeg -y
```

The separate Gradio/Transformers demos need additional dependencies and have
not been validated in this server environment:

```bash
python -m pip install gradio==5.44.1 gradio_client==1.12.1 accelerate
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
    --host 0.0.0.0 --port 8899
```

Open `http://<server-ip>:8899` in your browser.
The built-in GUI supports text, file upload (audio / image / video),
live microphone recording with waveform visualization, and real-time
token-by-token streaming output.

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
