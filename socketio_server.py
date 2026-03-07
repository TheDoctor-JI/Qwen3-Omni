#!/usr/bin/env python3
"""
Qwen3-Omni Socket.IO Streaming Server
======================================

Loads Qwen3-Omni via vLLM (identical configuration to web_demo.py) with full
multimodal support, then exposes a Socket.IO interface for real-time streaming.
A built-in interactive web GUI is served at the root URL.

Additional requirements beyond web_demo.py:
    pip install "python-socketio[asyncio_client]" aiohttp

Usage:
    python socketio_server.py \\
        --checkpoint-path ./Qwen3-Omni-30B-A3B-Thinking \\
        --host 127.0.0.1 --port 8902

    Then open  http://127.0.0.1:8902  in your browser.

Socket.IO protocol
------------------
  Client → Server events:
    "generate"  { messages: [{role, content: [{type, ...}]}],
                  params:{temperature,top_p,top_k,
                  max_tokens,thinking_mode},
                  request_id? }   — request_id is optional; server generates one if absent
    "stop"      { request_id? }  — cancel the running generation

  Server → Client events:
    "server_ready"         { sid }
    "generation_start"     { request_id }
    "token"                { request_id, delta, full_text, num_tokens,
                             elapsed, ttft, finished }
    "generation_complete"  { request_id, full_text, total_time,
                             num_tokens, tokens_per_second, ttft }
    "generation_stopped"   { request_id, partial_text }
    "generation_error"     { request_id, error }

Message format (inside "messages"):
    Each message is {role: str, content: [{type, ...}]}.
    Text items:  {type: "text", text: "..."}
    Audio items: {type: "audio", data: "<base64>", suffix: ".wav"}
    Image items: {type: "image", data: "<base64>", suffix: ".jpg"}
    Video items: {type: "video", data: "<base64>", suffix: ".mp4"}
    Roles: "system", "user", "assistant", "tool"
"""

import asyncio
import base64
import os
import tempfile
import time
import uuid

import torch
import yaml

os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from argparse import ArgumentParser
import socketio
from aiohttp import web
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor


DEFAULT_CKPT_PATH = "./Qwen3-Omni-30B-A3B-Thinking"

# ---------------------------------------------------------------------------
# Model loading  (identical to web_demo.py)
# ---------------------------------------------------------------------------

def _load_config(config_path):
    """Load YAML config file. Returns empty dict if path is None or missing."""
    if not config_path or not os.path.isfile(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def _load_model_processor(args):
    config = _load_config(getattr(args, 'config', None))

    from vllm import AsyncLLMEngine, AsyncEngineArgs
    engine_args = AsyncEngineArgs(
        model=args.checkpoint_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=torch.cuda.device_count(),
        limit_mm_per_prompt={'image': 1, 'video': 5, 'audio': 10},
        max_num_seqs=1,
        max_model_len=65535,
        seed=1234,
    )
    model = AsyncLLMEngine.from_engine_args(engine_args)
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.checkpoint_path)

    # Override max audio duration from config (default: 300s = 5 min)
    audio_cfg = config.get('audio', {})
    max_audio_sec = audio_cfg.get('max_audio_duration_sec', None)
    if max_audio_sec is not None:
        max_audio_sec = int(max_audio_sec)
        fe = processor.feature_extractor
        fe.n_samples = max_audio_sec * fe.sampling_rate
        fe.nb_max_frames = fe.n_samples // fe.hop_length
        print(f"[*] Audio max duration overridden to {max_audio_sec}s "
              f"(n_samples={fe.n_samples}, nb_max_frames={fe.nb_max_frames})")

    return model, processor


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------

def _save_temp_b64(data_b64, suffix):
    """Decode a base64 blob and write it to a temp file; return its path."""
    raw = base64.b64decode(data_b64)
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, 'wb') as fh:
        fh.write(raw)
    return path


def _build_messages(payload):
    """
    Convert a socket payload into (messages_list, temp_file_paths).

    The payload uses the messages-based protocol where every turn is a
    structured message with inline base64 media.  This function
    materialises any base64 media blobs to temp files (so that
    process_mm_info / apply_chat_template can reference file paths),
    and returns a messages list ready for the model pipeline.

    Caller is responsible for deleting temp files afterwards.
    """
    messages = []
    temp_files = []

    default_suffixes = {"audio": ".wav", "image": ".jpg", "video": ".mp4"}

    for msg in payload.get("messages") or []:
        role = msg.get("role", "user")
        raw_content = msg.get("content")

        # Normalise content to list-of-dicts
        if isinstance(raw_content, str):
            raw_content = [{"type": "text", "text": raw_content}]
        if not isinstance(raw_content, list):
            continue

        out_content = []
        for item in raw_content:
            item_type = item.get("type", "text")

            if item_type in default_suffixes and item.get("data"):
                # Inline base64 media — materialise to temp file
                suffix = item.get("suffix") or default_suffixes[item_type]
                path = _save_temp_b64(item["data"], suffix)
                temp_files.append(path)
                out_content.append({"type": item_type, item_type: path})

            elif item_type in default_suffixes and item.get(item_type):
                # Already a file path (e.g. local testing) — pass through
                out_content.append(item)

            elif item_type == "text":
                out_content.append({"type": "text", "text": item.get("text", "")})

            else:
                # Unknown item type — pass through
                out_content.append(item)

        if out_content:
            messages.append({"role": role, "content": out_content})

    return messages, temp_files


# ---------------------------------------------------------------------------
# Core streaming generation coroutine
# ---------------------------------------------------------------------------

async def _stream_generate(sio, sid, model, processor, payload):
    from vllm import SamplingParams

    request_id = payload.get('request_id') or str(uuid.uuid4())
    p = payload.get("params") or {}
    temperature   = float(p.get("temperature",  0.7))
    top_p         = float(p.get("top_p",        0.95))
    top_k         = int  (p.get("top_k",        20))
    max_tokens    = int  (p.get("max_tokens",   16384))
    thinking_mode = bool (p.get("thinking_mode", True))

    messages, temp_files = _build_messages(payload)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )

    try:
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking_mode,
        )
    except TypeError:
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = {
        "prompt": prompt_text,
        "multi_modal_data": {},
        "mm_processor_kwargs": {"use_audio_in_video": True},
    }
    if images is not None: inputs["multi_modal_data"]["image"] = images
    if videos is not None: inputs["multi_modal_data"]["video"] = videos
    if audios is not None: inputs["multi_modal_data"]["audio"] = audios

    await sio.emit("generation_start", {"request_id": request_id}, to=sid)

    t_start   = time.perf_counter()
    prev_text = ""
    n_tokens  = 0
    ttft      = None

    try:
        async for output in model.generate(inputs, sampling_params, request_id):
            full_text = output.outputs[0].text
            delta = full_text[len(prev_text):]
            if delta:
                elapsed = time.perf_counter() - t_start
                if ttft is None:
                    ttft = elapsed
                n_tokens += 1
                await sio.emit("token", {
                    "request_id": request_id,
                    "delta":      delta,
                    "full_text":  full_text,
                    "num_tokens": n_tokens,
                    "elapsed":    round(elapsed, 3),
                    "ttft":       round(ttft, 3),
                    "finished":   output.finished,
                }, to=sid)
                prev_text = full_text

        total_time = time.perf_counter() - t_start
        tps = n_tokens / total_time if total_time > 0 else 0.0
        await sio.emit("generation_complete", {
            "request_id":        request_id,
            "full_text":         prev_text,
            "total_time":        round(total_time,  3),
            "num_tokens":        n_tokens,
            "tokens_per_second": round(tps, 1),
            "ttft":              round(ttft, 3) if ttft is not None else None,
        }, to=sid)

    except asyncio.CancelledError:
        await sio.emit("generation_stopped", {
            "request_id":  request_id,
            "partial_text": prev_text,
        }, to=sid)

    except Exception as exc:
        await sio.emit("generation_error", {
            "request_id": request_id,
            "error":      str(exc),
        }, to=sid)

    finally:
        for path in temp_files:
            try:
                os.unlink(path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Socket.IO application factory
# ---------------------------------------------------------------------------

def create_socketio_app(model, processor):
    sio = socketio.AsyncServer(
        async_mode="aiohttp",
        cors_allowed_origins="*",
        max_http_buffer_size=100 * 1024 * 1024,   # 100 MB – enough for audio/video
    )
    app = web.Application()
    sio.attach(app)

    _active = {}   # sid -> (asyncio.Task, request_id: str)

    async def _abort_active(sid, reason: str):
        """Cancel the active task for *sid* and abort the vLLM request.

        Calls model.abort(request_id) to immediately stop engine-side token
        generation, then cancels the asyncio task so CancelledError propagates
        and the server emits generation_stopped.  Abort failures are logged
        but never raised — cancellation is solely client-driven.
        """
        entry = _active.pop(sid, None)
        if entry is None:
            return
        task, req_id = entry
        if task and not task.done():
            try:
                await model.abort(req_id)
            except Exception as e:
                print(f"[!] model.abort({req_id}) failed ({reason}): {e}")
            task.cancel()
            print(f"[x] {reason}: cancelled task for {sid} (request_id={req_id})")

    @sio.on("connect")
    async def on_connect(sid, environ):
        print(f"[+] connect    {sid}")
        await sio.emit("server_ready", {"sid": sid}, to=sid)

    @sio.on("disconnect")
    async def on_disconnect(sid):
        print(f"[-] disconnect {sid}")
        await _abort_active(sid, "disconnect")

    @sio.on("generate")
    async def on_generate(sid, payload):
        # Cancel any in-flight generation before starting a new one
        await _abort_active(sid, "new generate")
        request_id = payload.get('request_id') or str(uuid.uuid4())
        # Inject the resolved request_id back so _stream_generate uses it
        payload['request_id'] = request_id
        task = asyncio.create_task(
            _stream_generate(sio, sid, model, processor, payload)
        )
        _active[sid] = (task, request_id)

    @sio.on("stop")
    async def on_stop(sid, payload=None):
        await _abort_active(sid, "stop")

    async def handle_index(request):
        return web.Response(text=_GUI_HTML, content_type="text/html")

    app.router.add_get("/", handle_index)
    return app


# ---------------------------------------------------------------------------
# Embedded interactive GUI
# ---------------------------------------------------------------------------

_GUI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Qwen3-Omni · Live Stream</title>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<style>
:root {
  --bg:           #0d0d0f;
  --surface:      #18181b;
  --surface2:     #24242a;
  --border:       #2e2e38;
  --accent:       #7c6af7;
  --accent2:      #5eead4;
  --text:         #e4e4e7;
  --muted:        #71717a;
  --think-bg:     #13132a;
  --think-border: #4a4a8a;
  --think-text:   #a5b4fc;
  --green:        #22c55e;
  --red:          #ef4444;
  --radius:       10px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ── Header ─────────────────────────────────────────────────────────────── */
header {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 20px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
}
header .logo { font-size: 1.25rem; }
header h1 { font-size: 1rem; font-weight: 600; }
#conn-badge {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: .78rem;
  color: var(--muted);
}
#conn-dot {
  width: 9px; height: 9px;
  border-radius: 50%;
  background: var(--red);
  transition: background .3s;
}
#conn-dot.ok { background: var(--green); }

/* ── Main layout ─────────────────────────────────────────────────────────── */
.layout {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
.sidebar {
  width: 270px;
  flex-shrink: 0;
  background: var(--surface);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 0;
  overflow-y: auto;
  padding-bottom: 12px;
}
.sb-section {
  padding: 14px 16px 10px;
  border-bottom: 1px solid var(--border);
}
.sb-title {
  font-size: .72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .07em;
  color: var(--muted);
  margin-bottom: 10px;
}
label.field-label {
  display: block;
  font-size: .78rem;
  color: var(--muted);
  margin-bottom: 3px;
  margin-top: 9px;
}
label.field-label:first-of-type { margin-top: 0; }
textarea.sys-prompt {
  width: 100%;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  padding: 8px 10px;
  font-size: .82rem;
  resize: vertical;
  min-height: 72px;
  line-height: 1.5;
}
textarea.sys-prompt:focus { outline: none; border-color: var(--accent); }

/* sliders */
.slider-row { display: flex; align-items: center; gap: 8px; }
.slider-row input[type=range] {
  flex: 1;
  accent-color: var(--accent);
  height: 4px;
}
.slider-val {
  min-width: 38px;
  text-align: right;
  font-size: .8rem;
  color: var(--accent);
  font-variant-numeric: tabular-nums;
}
.toggle-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: .82rem;
  cursor: pointer;
  padding-top: 4px;
}
.toggle-row input[type=checkbox] {
  accent-color: var(--accent);
  width: 15px; height: 15px;
  cursor: pointer;
}

/* file drops */
.file-drop-wrap { position: relative; margin-bottom: 4px; }
.file-drop {
  position: relative;
  border: 2px dashed var(--border);
  border-radius: 7px;
  padding: 9px 10px;
  font-size: .78rem;
  color: var(--muted);
  cursor: pointer;
  transition: border-color .2s, background .2s;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}
.file-drop:hover { border-color: var(--accent); background: rgba(124,106,247,.06); }
.file-drop.loaded { border-color: var(--accent2); color: var(--accent2); }
.file-drop input[type=file] {
  position: absolute; inset: 0;
  opacity: 0; cursor: pointer; width: 100%; height: 100%;
}
.file-clear {
  font-size: .72rem;
  color: var(--red);
  cursor: pointer;
  display: none;
  margin-top: 2px;
}
.file-clear.visible { display: block; }

/* ── Audio recorder ─────────────────────────────────────────────────────── */
.audio-section { margin-bottom: 4px; }
.audio-tabs { display: flex; gap: 4px; margin-bottom: 6px; }
.atab {
  flex: 1;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 5px;
  color: var(--muted);
  font-size: .75rem;
  font-weight: 600;
  padding: 4px 0;
  cursor: pointer;
  transition: background .15s, color .15s;
}
.atab:hover { background: var(--border); color: var(--text); }
.atab.active { background: var(--accent); border-color: var(--accent); color: #fff; }
#waveform {
  width: 100%;
  height: 48px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 6px;
  display: block;
}
.rec-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 6px;
}
.mic-btn {
  width: 38px; height: 38px;
  border-radius: 50%;
  border: 2px solid var(--border);
  background: var(--surface2);
  font-size: 1.15rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: border-color .15s, background .15s;
  flex-shrink: 0;
  line-height: 1;
}
.mic-btn:hover { border-color: var(--accent); }
.mic-btn.recording {
  border-color: var(--red);
  background: rgba(239,68,68,.15);
  animation: recPulse 1s ease-in-out infinite;
}
@keyframes recPulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,.5); }
  50%       { box-shadow: 0 0 0 7px rgba(239,68,68,0); }
}
.rec-timer {
  font-size: .85rem;
  font-variant-numeric: tabular-nums;
  color: var(--accent2);
  min-width: 38px;
}
.rec-status { font-size: .72rem; color: var(--muted); }
.rec-status.live { color: var(--red); font-weight: 600; }
#audio-preview {
  width: 100%;
  margin-top: 7px;
  display: none;
  accent-color: var(--accent);
  border-radius: 6px;
}
.rec-discard {
  font-size: .72rem;
  color: var(--red);
  cursor: pointer;
  display: none;
  margin-top: 3px;
}
.rec-discard.visible { display: block; }

.btn-clear-hist {
  margin: 14px 16px 0;
  width: calc(100% - 32px);
}

/* ── Chat column ─────────────────────────────────────────────────────────── */
.chat-col {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

#messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px 24px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

/* bubbles */
.msg-wrap { display: flex; flex-direction: column; }
.msg-wrap.user  { align-items: flex-end; }
.msg-wrap.asst  { align-items: flex-start; }
.bubble {
  max-width: 78%;
  border-radius: var(--radius);
  padding: 11px 15px;
  font-size: .9rem;
  line-height: 1.65;
  white-space: pre-wrap;
  word-break: break-word;
}
.msg-wrap.user .bubble {
  background: var(--accent);
  color: #fff;
}
.msg-wrap.asst .bubble {
  background: var(--surface2);
  border: 1px solid var(--border);
}

/* thinking block */
details.think-details {
  background: var(--think-bg);
  border: 1px solid var(--think-border);
  border-radius: 7px;
  margin-bottom: 8px;
  overflow: hidden;
}
details.think-details summary {
  list-style: none;
  padding: 7px 12px;
  font-size: .75rem;
  font-weight: 700;
  color: var(--think-border);
  text-transform: uppercase;
  letter-spacing: .07em;
  cursor: pointer;
  user-select: none;
}
details.think-details summary::marker,
details.think-details summary::-webkit-details-marker { display: none; }
details.think-details summary::before {
  content: '▶ ';
  transition: transform .2s;
  display: inline-block;
}
details[open].think-details summary::before { transform: rotate(90deg); }
.think-body {
  padding: 6px 14px 10px;
  font-size: .8rem;
  color: var(--think-text);
  font-style: italic;
  line-height: 1.6;
  white-space: pre-wrap;
  max-height: 220px;
  overflow-y: auto;
}

/* output text */
.output-text { }

/* streaming token flash */
@keyframes tokenFlash {
  0%   { background: rgba(124, 106, 247, .45); border-radius: 2px; }
  100% { background: transparent; }
}
.tok { animation: tokenFlash .5s ease-out forwards; }

/* cursor */
@keyframes blink { 50% { opacity: 0; } }
.cursor {
  display: inline-block;
  width: 2px; height: 1.05em;
  background: var(--accent);
  margin-left: 1px;
  vertical-align: text-bottom;
  animation: blink .75s step-end infinite;
  flex-shrink: 0;
}

/* ── Metrics bar ─────────────────────────────────────────────────────────── */
.metrics-bar {
  flex-shrink: 0;
  display: flex;
  gap: 22px;
  align-items: center;
  padding: 5px 20px;
  background: var(--surface);
  border-top: 1px solid var(--border);
  font-size: .75rem;
  color: var(--muted);
  min-height: 28px;
}
.metrics-bar .mv { color: var(--accent2); font-variant-numeric: tabular-nums; }
#gen-status {
  margin-left: auto;
  font-size: .72rem;
  display: flex;
  align-items: center;
  gap: 5px;
}
.spinner {
  width: 10px; height: 10px;
  border: 2px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin .6s linear infinite;
  display: none;
}
.spinner.active { display: block; }
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Input bar ───────────────────────────────────────────────────────────── */
.input-bar {
  flex-shrink: 0;
  display: flex;
  gap: 8px;
  align-items: flex-end;
  padding: 12px 20px;
  background: var(--surface);
  border-top: 1px solid var(--border);
}
#text-input {
  flex: 1;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text);
  padding: 10px 13px;
  font-size: .9rem;
  font-family: inherit;
  resize: none;
  min-height: 42px;
  max-height: 150px;
  line-height: 1.5;
  transition: border-color .15s;
}
#text-input:focus { outline: none; border-color: var(--accent); }

.btn {
  flex-shrink: 0;
  border: none;
  border-radius: 8px;
  padding: 0 18px;
  height: 42px;
  font-size: .85rem;
  font-weight: 600;
  cursor: pointer;
  transition: opacity .15s, transform .08s;
  white-space: nowrap;
}
.btn:hover  { opacity: .88; }
.btn:active { transform: scale(.96); }
.btn:disabled { opacity: .35; cursor: not-allowed; pointer-events: none; }
.btn-send  { background: var(--accent); color: #fff; }
.btn-stop  { background: var(--red);    color: #fff; display: none; }
.btn-clear { background: var(--surface2); color: var(--muted); border: 1px solid var(--border); }

/* error msg */
.err-msg {
  color: var(--red);
  font-size: .82rem;
  padding: 8px 14px;
  background: rgba(239,68,68,.09);
  border: 1px solid rgba(239,68,68,.35);
  border-radius: 7px;
  max-width: 78%;
}

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>

<!-- ── Header ──────────────────────────────────────────────────────────── -->
<header>
  <span class="logo">🤖</span>
  <h1>Qwen3-Omni · Live Stream</h1>
  <div id="conn-badge">
    <span id="conn-label">disconnected</span>
    <div id="conn-dot"></div>
  </div>
</header>

<div class="layout">

  <!-- ── Sidebar ─────────────────────────────────────────────────────────── -->
  <aside class="sidebar">

    <div class="sb-section">
      <div class="sb-title">System Prompt</div>
      <textarea class="sys-prompt" id="system-prompt" placeholder="Optional system prompt…"></textarea>
    </div>

    <div class="sb-section">
      <div class="sb-title">Parameters</div>

      <label class="field-label">Temperature</label>
      <div class="slider-row">
        <input type="range" id="s-temp"   min="0.05" max="2.0"  step="0.05" value="0.7">
        <span  class="slider-val" id="v-temp">0.7</span>
      </div>

      <label class="field-label">Top P</label>
      <div class="slider-row">
        <input type="range" id="s-top-p"  min="0.05" max="1.0"  step="0.05" value="0.95">
        <span  class="slider-val" id="v-top-p">0.95</span>
      </div>

      <label class="field-label">Top K</label>
      <div class="slider-row">
        <input type="range" id="s-top-k"  min="1"    max="100"  step="1"    value="20">
        <span  class="slider-val" id="v-top-k">20</span>
      </div>

      <label class="field-label">Max Tokens</label>
      <div class="slider-row">
        <input type="range" id="s-maxtok" min="256"  max="16384" step="256" value="4096">
        <span  class="slider-val" id="v-maxtok">4096</span>
      </div>

      <label class="toggle-row" style="margin-top:10px">
        <input type="checkbox" id="thinking-mode" checked>
        Enable Thinking Mode
      </label>
    </div>

    <div class="sb-section">
      <div class="sb-title">Media Attachments</div>

      <!-- Audio: tabbed upload / live record -->
      <div class="audio-section">
        <div class="audio-tabs">
          <button class="atab active" id="atab-upload" onclick="switchAudioTab('upload')">📁 Upload</button>
          <button class="atab"        id="atab-record" onclick="switchAudioTab('record')">🎙 Record</button>
        </div>

        <!-- Upload pane -->
        <div id="apane-upload">
          <div class="file-drop-wrap">
            <div class="file-drop" id="drop-audio">
              🎵 Audio (.wav / .mp3 / .ogg)
              <input type="file" id="file-audio" accept="audio/*">
            </div>
            <span class="file-clear" id="clear-audio" onclick="clearFile('audio')">✕ remove</span>
          </div>
        </div>

        <!-- Record pane -->
        <div id="apane-record" style="display:none">
          <canvas id="waveform" width="256" height="48"></canvas>
          <div class="rec-row">
            <button class="mic-btn" id="mic-btn" onclick="toggleRecording()" title="Start recording">🎙</button>
            <span class="rec-timer" id="rec-timer">00:00</span>
            <span class="rec-status" id="rec-status">Ready</span>
          </div>
          <audio id="audio-preview" controls></audio>
          <span class="rec-discard" id="rec-discard" onclick="clearRecording()">✕ discard recording</span>
        </div>
      </div>

      <!-- Image -->
      <div class="file-drop-wrap">
        <div class="file-drop" id="drop-image">
          🖼 Image (.jpg / .png / .webp)
          <input type="file" id="file-image" accept="image/*">
        </div>
        <span class="file-clear" id="clear-image" onclick="clearFile('image')">✕ remove</span>
      </div>

      <!-- Video -->
      <div class="file-drop-wrap">
        <div class="file-drop" id="drop-video">
          🎬 Video (.mp4 / .webm)
          <input type="file" id="file-video" accept="video/*">
        </div>
        <span class="file-clear" id="clear-video" onclick="clearFile('video')">✕ remove</span>
      </div>
    </div>

    <button class="btn btn-clear btn-clear-hist" onclick="clearHistory()">
      Clear Chat History
    </button>
  </aside>

  <!-- ── Chat column ──────────────────────────────────────────────────────── -->
  <div class="chat-col">

    <div id="messages"></div>

    <!-- ── Metrics bar ──────────────────────────────────────────────────── -->
    <div class="metrics-bar">
      <span>TTFT&nbsp;<span class="mv" id="m-ttft">—</span></span>
      <span>Tokens&nbsp;<span class="mv" id="m-tokens">—</span></span>
      <span>TPS&nbsp;<span class="mv" id="m-tps">—</span></span>
      <span>Total&nbsp;<span class="mv" id="m-total">—</span></span>
      <div id="gen-status">
        <div class="spinner" id="spinner"></div>
        <span id="status-text"></span>
      </div>
    </div>

    <!-- ── Input bar ────────────────────────────────────────────────────── -->
    <div class="input-bar">
      <textarea id="text-input"
        placeholder="Type a message… (Enter to send · Shift+Enter for new line)"
        rows="1"></textarea>
      <button class="btn btn-send" id="btn-send" onclick="sendMessage()">Send</button>
      <button class="btn btn-stop" id="btn-stop" onclick="stopGeneration()">⏹ Stop</button>
    </div>

  </div><!-- .chat-col -->
</div><!-- .layout -->

<script>
// ── State ─────────────────────────────────────────────────────────────────
const SERVER = window.location.origin;
let socket = null;
let generating = false;
let history = [];          // [{role, content}] – text-only for multi-turn

// Streaming DOM state
let asstWrap  = null;   // current assistant .msg-wrap
let thinkBody = null;   // <div class="think-body"> inside the details
let thinkDet  = null;   // <details class="think-details">
let outputEl  = null;   // <span class="output-text">
let cursorEl  = null;
let fullBuf   = '';
let rendThink = 0;
let rendOut   = 0;
let tStart    = 0;

// Pending file payloads: {data, suffix, name}
const pending = { audio: null, image: null, video: null };

// ── Mic recorder state ────────────────────────────────────────────────────
let mediaRecorder  = null;
let audioChunks    = [];
let recStream      = null;
let recTimerSec    = 0;
let recTimerHandle = null;
let audioCtx       = null;
let analyser       = null;
let waveAnimHandle = null;

// ── Connect ───────────────────────────────────────────────────────────────
function connect() {
  socket = io(SERVER, { transports: ['websocket'] });

  socket.on('connect', () => {
    document.getElementById('conn-dot').classList.add('ok');
    document.getElementById('conn-label').textContent = 'connected';
    document.getElementById('btn-send').disabled = false;
  });

  socket.on('disconnect', () => {
    document.getElementById('conn-dot').classList.remove('ok');
    document.getElementById('conn-label').textContent = 'disconnected';
    document.getElementById('btn-send').disabled = true;
    if (generating) finalizeAsst(null);
  });

  socket.on('server_ready', () => {
    setStatus('ready');
  });

  socket.on('generation_start', () => {
    tStart = performance.now();
    setMetrics('…', 0, '…', '…');
    setStatus('generating…');
  });

  socket.on('token', (d) => {
    appendDelta(d.delta);
    const elapsedSec = (performance.now() - tStart) / 1000;
    const tps = elapsedSec > 0 ? (d.num_tokens / elapsedSec).toFixed(1) : '…';
    setMetrics(d.ttft + 's', d.num_tokens, tps, elapsedSec.toFixed(2) + 's');
  });

  socket.on('generation_complete', (d) => {
    finalizeAsst(d.full_text);
    setMetrics(
      d.ttft != null ? d.ttft + 's' : '—',
      d.num_tokens,
      d.tokens_per_second + ' t/s',
      d.total_time + 's'
    );
    setStatus('done');
    history.push({ role: 'assistant', content: [{type: 'text', text: d.full_text}] });
    setGenerating(false);
  });

  socket.on('generation_stopped', (d) => {
    finalizeAsst(d.partial_text || null);
    setStatus('stopped');
    if (d.partial_text) history.push({ role: 'assistant', content: [{type: 'text', text: d.partial_text}] });
    setGenerating(false);
  });

  socket.on('generation_error', (d) => {
    finalizeAsst(null);
    showError(d.error);
    setStatus('error');
    setGenerating(false);
  });
}

// ── UI helpers ────────────────────────────────────────────────────────────
function setGenerating(v) {
  generating = v;
  document.getElementById('btn-send').style.display = v ? 'none' : '';
  document.getElementById('btn-stop').style.display = v ? ''     : 'none';
  document.getElementById('spinner').classList.toggle('active', v);
}

function setStatus(msg) {
  document.getElementById('status-text').textContent = msg;
}

function setMetrics(ttft, tokens, tps, total) {
  document.getElementById('m-ttft').textContent   = ttft;
  document.getElementById('m-tokens').textContent = tokens;
  document.getElementById('m-tps').textContent    = tps;
  document.getElementById('m-total').textContent  = total;
}

function scrollDown() {
  const el = document.getElementById('messages');
  el.scrollTop = el.scrollHeight;
}

function showError(msg) {
  const el = document.createElement('div');
  el.className = 'err-msg';
  el.textContent = '⚠ ' + msg;
  document.getElementById('messages').appendChild(el);
  scrollDown();
}

// ── Streamed assistant bubble ─────────────────────────────────────────────
function startAsstBubble() {
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap asst';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  // Thinking block (hidden until <think> appears)
  thinkDet = document.createElement('details');
  thinkDet.className = 'think-details';
  thinkDet.open = true;
  thinkDet.style.display = 'none';

  const summary = document.createElement('summary');
  summary.textContent = '💭 Thinking…';
  thinkDet.appendChild(summary);

  thinkBody = document.createElement('div');
  thinkBody.className = 'think-body';
  thinkDet.appendChild(thinkBody);

  // Output text span
  outputEl = document.createElement('span');
  outputEl.className = 'output-text';

  // Blinking cursor
  cursorEl = document.createElement('span');
  cursorEl.className = 'cursor';

  bubble.appendChild(thinkDet);
  bubble.appendChild(outputEl);
  bubble.appendChild(cursorEl);
  wrap.appendChild(bubble);

  document.getElementById('messages').appendChild(wrap);
  asstWrap = wrap;
  scrollDown();
}

// Parse full accumulated buffer into {thinkText, outputText, thinkOpen, thinkDone}
function parseBuf(buf) {
  const open  = buf.indexOf('<think>');
  const close = buf.indexOf('</think>');
  if (open === -1) {
    return { thinkText: '', outputText: buf, thinkOpen: false, thinkDone: false };
  }
  const thinkText = buf.slice(open + 7, close !== -1 ? close : buf.length);
  const outputText = close !== -1 ? buf.slice(close + 8) : '';
  return { thinkText, outputText, thinkOpen: true, thinkDone: close !== -1 };
}

// Append a span of text to an element with the flash animation
function appendFlash(el, text) {
  if (!text) return;
  const span = document.createElement('span');
  span.className = 'tok';
  span.textContent = text;
  el.appendChild(span);
  span.addEventListener('animationend', () => span.classList.remove('tok'), { once: true });
}

function appendDelta(delta) {
  if (!asstWrap) startAsstBubble();

  fullBuf += delta;
  const { thinkText, outputText, thinkOpen, thinkDone } = parseBuf(fullBuf);

  // Render newly arrived thinking chars
  if (thinkOpen && thinkText.length > rendThink) {
    thinkDet.style.display = '';
    appendFlash(thinkBody, thinkText.slice(rendThink));
    rendThink = thinkText.length;
    thinkBody.scrollTop = thinkBody.scrollHeight;
    // When thinking is done collapse it so output gets focus
    if (thinkDone) {
      thinkDet.open = false;
      thinkDet.querySelector('summary').textContent = '💭 Thinking (click to expand)';
    }
  }

  // Render newly arrived output chars
  if (outputText.length > rendOut) {
    // Insert flash span before the cursor
    const newChars = outputText.slice(rendOut);
    rendOut = outputText.length;
    const span = document.createElement('span');
    span.className = 'tok';
    span.textContent = newChars;
    outputEl.insertBefore(span, null); // append to outputEl
    span.addEventListener('animationend', () => span.classList.remove('tok'), { once: true });
  }

  scrollDown();
}

function finalizeAsst(fullText) {
  if (cursorEl) { cursorEl.remove(); cursorEl = null; }
  // Reset state for next turn
  asstWrap = null; thinkBody = null; thinkDet = null; outputEl = null;
  fullBuf = ''; rendThink = 0; rendOut = 0;
}

// ── User bubble ───────────────────────────────────────────────────────────
function addUserBubble(text, files) {
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap user';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  let label = '';
  if (files.audio) label += '[🎵 audio] ';
  if (files.image) label += '[🖼 image] ';
  if (files.video) label += '[🎬 video] ';
  bubble.textContent = (label + text).trim();
  wrap.appendChild(bubble);
  document.getElementById('messages').appendChild(wrap);
  scrollDown();
}

// ── File handling ─────────────────────────────────────────────────────────
const FILE_META = {
  audio: { icon: '🎵', label: 'Audio (.wav / .mp3 / .ogg)', accept: 'audio/*' },
  image: { icon: '🖼', label: 'Image (.jpg / .png / .webp)', accept: 'image/*' },
  video: { icon: '🎬', label: 'Video (.mp4 / .webm)',        accept: 'video/*' },
};

function setupFileInput(type) {
  document.getElementById('file-' + type).addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const b64 = ev.target.result.split(',')[1];
      const suffix = '.' + (file.name.split('.').pop() || type);
      pending[type] = { data: b64, suffix, name: file.name };
      const drop = document.getElementById('drop-' + type);
      drop.classList.add('loaded');
      // Replace text node while keeping the hidden file input
      drop.childNodes[0].textContent = FILE_META[type].icon + ' ' + file.name + ' ';
      document.getElementById('clear-' + type).classList.add('visible');
    };
    reader.readAsDataURL(file);
  });
}

function clearFile(type) {
  pending[type] = null;
  // Reset drop label
  const drop = document.getElementById('drop-' + type);
  drop.classList.remove('loaded');
  drop.childNodes[0].textContent = FILE_META[type].icon + ' ' + FILE_META[type].label + ' ';
  // Reset input so same file can be re-selected
  document.getElementById('file-' + type).value = '';
  document.getElementById('clear-' + type).classList.remove('visible');
}

// Clear audio from whichever tab supplied it, used after send
function clearAudioAfterSend() {
  pending.audio = null;
  // Reset upload pane
  const drop = document.getElementById('drop-audio');
  drop.classList.remove('loaded');
  drop.childNodes[0].textContent = FILE_META.audio.icon + ' ' + FILE_META.audio.label + ' ';
  document.getElementById('file-audio').value = '';
  document.getElementById('clear-audio').classList.remove('visible');
  // Reset record pane (non-destructively — don't stop an active recording)
  if (!mediaRecorder || mediaRecorder.state === 'inactive') clearRecording();
}

// ── Audio recorder ────────────────────────────────────────────────────────
function switchAudioTab(tab) {
  document.getElementById('apane-upload').style.display = tab === 'upload' ? ''     : 'none';
  document.getElementById('apane-record').style.display = tab === 'record' ? ''     : 'none';
  document.getElementById('atab-upload').classList.toggle('active', tab === 'upload');
  document.getElementById('atab-record').classList.toggle('active', tab === 'record');
  // Stop any active recording when leaving the tab
  if (tab !== 'record' && mediaRecorder && mediaRecorder.state === 'recording') stopRecording();
}

function toggleRecording() {
  if (!mediaRecorder || mediaRecorder.state === 'inactive') {
    startRecording();
  } else {
    stopRecording();
  }
}

async function startRecording() {
  clearRecording();   // discard any previous take
  try {
    recStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    showError('Microphone access denied: ' + e.message);
    return;
  }

  // Waveform analyser
  audioCtx  = new (window.AudioContext || window.webkitAudioContext)();
  const src = audioCtx.createMediaStreamSource(recStream);
  analyser  = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  src.connect(analyser);
  drawWaveform();

  audioChunks = [];
  const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus'
             : MediaRecorder.isTypeSupported('audio/webm')             ? 'audio/webm'
             : '';
  mediaRecorder = new MediaRecorder(recStream, mime ? { mimeType: mime } : {});
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
  mediaRecorder.onstop = onRecordingStop;
  mediaRecorder.start(100);

  // UI
  document.getElementById('mic-btn').classList.add('recording');
  document.getElementById('mic-btn').textContent    = '⏹';
  document.getElementById('mic-btn').title          = 'Stop recording';
  document.getElementById('rec-status').textContent = '● Recording';
  document.getElementById('rec-status').classList.add('live');
  document.getElementById('audio-preview').style.display = 'none';
  document.getElementById('rec-discard').classList.remove('visible');

  // Timer
  recTimerSec = 0;
  document.getElementById('rec-timer').textContent = '00:00';
  recTimerHandle = setInterval(() => {
    recTimerSec++;
    document.getElementById('rec-timer').textContent = recFmtTime(recTimerSec);
  }, 1000);
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  if (recTimerHandle) { clearInterval(recTimerHandle); recTimerHandle = null; }
  if (waveAnimHandle) { cancelAnimationFrame(waveAnimHandle); waveAnimHandle = null; }
  if (recStream)  { recStream.getTracks().forEach(t => t.stop()); recStream = null; }
  if (audioCtx)   { audioCtx.close(); audioCtx = null; analyser = null; }
  document.getElementById('mic-btn').classList.remove('recording');
  document.getElementById('mic-btn').textContent    = '🎙';
  document.getElementById('mic-btn').title          = 'Start recording';
  document.getElementById('rec-status').textContent = 'Processing…';
  document.getElementById('rec-status').classList.remove('live');
  const canvas = document.getElementById('waveform');
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

function onRecordingStop() {
  const mimeType = (mediaRecorder && mediaRecorder.mimeType) || 'audio/webm';
  const suffix   = mimeType.includes('ogg') ? '.ogg' : '.webm';
  const blob     = new Blob(audioChunks, { type: mimeType });
  const url      = URL.createObjectURL(blob);

  const preview = document.getElementById('audio-preview');
  if (preview.src) URL.revokeObjectURL(preview.src);
  preview.src             = url;
  preview.style.display   = 'block';

  const reader = new FileReader();
  reader.onload = ev => {
    const b64 = ev.target.result.split(',')[1];
    pending.audio = { data: b64, suffix, name: 'recording' + suffix };
    document.getElementById('rec-status').textContent = '✔ Ready  (' + recFmtTime(recTimerSec) + ')';
    document.getElementById('rec-discard').classList.add('visible');
  };
  reader.readAsDataURL(blob);
}

function clearRecording() {
  pending.audio = null;
  audioChunks   = [];
  const preview = document.getElementById('audio-preview');
  if (preview && preview.src) { URL.revokeObjectURL(preview.src); preview.removeAttribute('src'); }
  if (preview) preview.style.display = 'none';
  const disc   = document.getElementById('rec-discard');
  if (disc)    disc.classList.remove('visible');
  const status = document.getElementById('rec-status');
  if (status)  { status.textContent = 'Ready'; status.classList.remove('live'); }
  recTimerSec = 0;
  const timer = document.getElementById('rec-timer');
  if (timer)   timer.textContent = '00:00';
}

function recFmtTime(sec) {
  return String(Math.floor(sec / 60)).padStart(2, '0') + ':' +
         String(sec % 60).padStart(2, '0');
}

function drawWaveform() {
  if (!analyser) return;
  const canvas = document.getElementById('waveform');
  // Match internal resolution to CSS size once
  canvas.width  = canvas.clientWidth  || 256;
  canvas.height = canvas.clientHeight || 48;
  const ctx    = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const bufLen = analyser.frequencyBinCount;
  const data   = new Uint8Array(bufLen);
  const bg     = getComputedStyle(document.documentElement).getPropertyValue('--surface2').trim();
  const fg     = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();

  function render() {
    if (!analyser) return;
    waveAnimHandle = requestAnimationFrame(render);
    analyser.getByteTimeDomainData(data);
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);
    ctx.lineWidth   = 1.8;
    ctx.strokeStyle = fg;
    ctx.beginPath();
    const sliceW = W / bufLen;
    let x = 0;
    for (let i = 0; i < bufLen; i++) {
      const y = (data[i] / 128.0) * (H / 2);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      x += sliceW;
    }
    ctx.lineTo(W, H / 2);
    ctx.stroke();
  }
  render();
}

// ── Send & Stop ───────────────────────────────────────────────────────────
function sendMessage() {
  if (generating) return;
  if (!socket || !socket.connected) return;

  const ta   = document.getElementById('text-input');
  const text = ta.value.trim();
  const hasFiles = { audio: !!pending.audio, image: !!pending.image, video: !!pending.video };
  if (!text && !hasFiles.audio && !hasFiles.image && !hasFiles.video) return;

  // Build user content items
  const userContent = [];
  if (pending.audio) userContent.push({type: 'audio', data: pending.audio.data, suffix: pending.audio.suffix || '.wav'});
  if (pending.image) userContent.push({type: 'image', data: pending.image.data, suffix: pending.image.suffix || '.jpg'});
  if (pending.video) userContent.push({type: 'video', data: pending.video.data, suffix: pending.video.suffix || '.mp4'});
  if (text) userContent.push({type: 'text', text});

  // Push to history
  history.push({role: 'user', content: userContent});

  addUserBubble(text, hasFiles);

  ta.value = '';
  ta.style.height = '42px';

  setGenerating(true);

  // Build messages: system prompt + full history (including current user turn)
  const messages = [];
  const sysPrompt = document.getElementById('system-prompt').value.trim();
  if (sysPrompt) messages.push({role: 'system', content: [{type: 'text', text: sysPrompt}]});
  for (const h of history) messages.push(h);

  const payload = {
    messages,
    params: {
      temperature:   parseFloat(document.getElementById('s-temp').value),
      top_p:         parseFloat(document.getElementById('s-top-p').value),
      top_k:         parseInt  (document.getElementById('s-top-k').value),
      max_tokens:    parseInt  (document.getElementById('s-maxtok').value),
      thinking_mode: document.getElementById('thinking-mode').checked,
    },
  };

  // Clear attached files after send
  if (pending.audio) clearAudioAfterSend();
  if (pending.image) clearFile('image');
  if (pending.video) clearFile('video');

  socket.emit('generate', payload);
}

function stopGeneration() {
  if (socket && socket.connected) socket.emit('stop', {});
}

function clearHistory() {
  history = [];
  document.getElementById('messages').innerHTML = '';
  setMetrics('—', '—', '—', '—');
  setStatus('');
}

// ── Slider labels ─────────────────────────────────────────────────────────
[['s-temp','v-temp'], ['s-top-p','v-top-p'], ['s-top-k','v-top-k'], ['s-maxtok','v-maxtok']]
  .forEach(([sid, vid]) => {
    const s = document.getElementById(sid), v = document.getElementById(vid);
    s.addEventListener('input', () => { v.textContent = s.value; });
  });

// ── Textarea auto-grow ────────────────────────────────────────────────────
const ta = document.getElementById('text-input');
ta.addEventListener('input', () => {
  ta.style.height = '42px';
  ta.style.height = Math.min(ta.scrollHeight, 150) + 'px';
});
ta.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// ── Boot ──────────────────────────────────────────────────────────────────
['audio', 'image', 'video'].forEach(setupFileInput);
document.getElementById('btn-send').disabled = true;
connect();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _get_args():
    parser = ArgumentParser(
        description="Qwen3-Omni Socket.IO streaming server with built-in interactive GUI"
    )
    parser.add_argument(
        "-c", "--checkpoint-path",
        default=DEFAULT_CKPT_PATH,
        help="Model checkpoint path (default: %(default)r)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (e.g. mm_llm_config.yaml) for runtime overrides",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Bind host (default: 127.0.0.1; use 0.0.0.0 to expose on LAN)",
    )
    parser.add_argument(
        "--port", type=int, default=8902,
        help="Bind port (default: 8902)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _get_args()
    print(f"[*] Loading model from: {args.checkpoint_path}")
    model, processor = _load_model_processor(args)
    print(f"[*] Model loaded. Serving GUI + Socket.IO on http://{args.host}:{args.port}")
    app = create_socketio_app(model, processor)
    web.run_app(app, host=args.host, port=args.port)
