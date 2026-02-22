# Qwen3-Omni Web Demo — Setup Guide

## Requirements
- CUDA-capable GPU(s) with **≥79 GB** total VRAM (BF16, Instruct model)
- CUDA 12.4 driver
- Miniconda / Anaconda

---

## 1. Create conda environment

```bash
conda create -n qwen3omni python=3.11 -y
conda activate qwen3omni
```

## 2. Install PyTorch (CUDA 12.4)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
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

## 5. Install Gradio and audio dependencies

```bash
pip install gradio==5.44.1 gradio_client==1.12.1 soundfile==0.13.1
```

## 6. Install FlashAttention 2 (optional but recommended)

```bash
pip install -U flash-attn --no-build-isolation
```

## 7. Install ffmpeg

```bash
conda install -c conda-forge ffmpeg -y
```

## 8. Download the model

```bash
# Hugging Face
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir ./Qwen3-Omni-30B-A3B-Instruct

# Or ModelScope (Mainland China)
pip install -U modelscope
modelscope download --model Qwen/Qwen3-Omni-30B-A3B-Instruct --local_dir ./Qwen3-Omni-30B-A3B-Instruct
```

---

## Launch

### vLLM backend — streaming, fast (recommended)

```bash
python web_demo.py -c ./Qwen3-Omni-30B-A3B-Instruct \
    --server-port 8901 --server-name 0.0.0.0
```

### Transformers backend — streaming + audio output support

```bash
python web_demo.py -c ./Qwen3-Omni-30B-A3B-Instruct \
    --use-transformers --flash-attn2 \
    --server-port 8901 --server-name 0.0.0.0

# Add --generate-audio to also produce speech output
```

Open `http://<server-ip>:8901` in your browser.

---

## Features

| Feature | vLLM | Transformers |
|---|---|---|
| Token streaming | ✅ (`AsyncLLMEngine`) | ✅ (`TextIteratorStreamer`) |
| Thinking mode toggle | ✅ | ✅ |
| Audio (speech) output | ❌ | ✅ (`--generate-audio`) |
| Speed | Fast | Slower |

- **TTFT** and total generation time are printed to the terminal on each request.
- **Thinking mode** can be toggled on/off in the UI sidebar at runtime.
