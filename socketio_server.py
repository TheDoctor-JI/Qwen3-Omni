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
    "items_cached"         { request_id, items: [{item_id, meta_key}] }
                           — emitted after preprocessing, before generation_start.
                             Lists every MM item confirmed in the server-side
                             encoding cache (both newly encoded and cache hits).
    "generation_start"     { request_id }
    "token"                { request_id, delta, full_text, num_tokens,
           elapsed, ttft, finished,
                             [on finished=true only:
                  full_generation_latency_sec, tokens_per_second,
                              generation_duration, generated_tokens,
                              time_to_first_token] }
    "generation_complete"  { request_id, full_text, full_generation_latency_sec,
                 num_tokens, tokens_per_second, ttft,
                 generation_duration, generated_tokens,
                 time_to_first_token }
    "generation_stopped"   { request_id, partial_text,
           full_generation_latency_sec, num_tokens, tokens_per_second, ttft,
                 generation_duration, generated_tokens,
                 time_to_first_token }
    "generation_error"     { request_id, error }

  Timing semantics:
    ttft / time_to_first_token = time from generation trigger to first token.
    generation_duration = time from first token to generation end.
    full_generation_latency_sec = ttft + generation_duration (when ttft is available).
    legacy total_time is emitted for backward compatibility.

Message format (inside "messages"):
    Each message is {role: str, content: [{type, ...}]}.
    Text items:  {type: "text", text: "..."}
    Audio items: {type: "audio", data: "<base64>", suffix: ".wav"}
    Audio stub:  {type: "audio", item_id: "...", duration_ms: N, sample_rate: N}
                 — lightweight reference for items the client believes are
                   already in the server's encoding cache. No "data" key.
                   Server looks up the cached encoding; raises ValueError on miss.
    Image items: {type: "image", data: "<base64>", suffix: ".jpg"}
    Video items: {type: "video", data: "<base64>", suffix: ".mp4"}
    Roles: "system", "user", "assistant", "tool"
"""

import asyncio
import base64
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import yaml

# Module-level logger — replaced with a file-backed logger in __main__
_logger = logging.getLogger('socketio_server')

os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from argparse import ArgumentParser
import socketio
from aiohttp import web
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor


DEFAULT_CKPT_PATH = "./Qwen3-Omni-30B-A3B-Thinking"

# Set during _load_model_processor; guards how many *new* (uncached) mm items
# a single request may introduce.
_MAX_NEW_MM_PER_REQUEST: int = 20

# Set during _load_model_processor; used to gate thinking-mode behavior.
_MODEL_IS_INSTRUCT: bool = False


# ---------------------------------------------------------------------------
# Per-session multimodal encoding cache
# ---------------------------------------------------------------------------

class MmItemCache:
    """Caches the output of ``process_mm_info`` per (item_id, meta_key).

    Thread-safe: a global lock protects the two dicts, and a per-item lock
    prevents two threads from encoding the same unseen item concurrently.
    """

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str], Any] = {}
        self._item_locks: Dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    @staticmethod
    def meta_key_for(item: dict) -> str:
        """Derive a metadata fingerprint from an annotated content item."""
        dur = item.get('duration_ms', '?')
        sr = item.get('sample_rate', '?')
        return f"{dur}ms_{sr}hz"

    def get(self, item_id: str, meta_key: str) -> Optional[Any]:
        """Return the cached encoding or None."""
        with self._lock:
            return self._cache.get((item_id, meta_key))

    def get_or_encode(
        self,
        item_id: str,
        meta_key: str,
        encode_fn: Callable[[], Any],
    ) -> Tuple[Any, bool]:
        """Return ``(encoded_result, is_new)``.

        If the item is already cached, returns it immediately.  Otherwise
        acquires a per-item lock (so concurrent callers for the *same*
        unseen item wait instead of double-encoding), runs *encode_fn*,
        stores the result, and returns it.
        """
        # Fast path — already cached
        with self._lock:
            cached = self._cache.get((item_id, meta_key))
            if cached is not None:
                return cached, False
            # Get or create per-item lock
            if item_id not in self._item_locks:
                self._item_locks[item_id] = threading.Lock()
            item_lock = self._item_locks[item_id]

        # Slow path — hold per-item lock, double-check, then encode
        with item_lock:
            with self._lock:
                cached = self._cache.get((item_id, meta_key))
                if cached is not None:
                    return cached, False
            result = encode_fn()
            with self._lock:
                self._cache[(item_id, meta_key)] = result
            return result, True

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


# ---------------------------------------------------------------------------
# Model loading  (identical to web_demo.py)
# ---------------------------------------------------------------------------

def _load_config(config_path):
    """Load YAML config file. Returns empty dict if path is None or missing."""
    if not config_path or not os.path.isfile(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


# Global config mapping
SERVER_CONFIG = {}

def _load_model_processor(args):
    global SERVER_CONFIG
    config = _load_config(getattr(args, 'config', None))
    SERVER_CONFIG = config

    global _MODEL_IS_INSTRUCT
    ckpt_lower = str(args.checkpoint_path).lower()
    _MODEL_IS_INSTRUCT = 'instruct' in ckpt_lower

    model_cfg = config.get('model', {})
    max_num_seqs = int(model_cfg.get('max_num_seqs', 2))
    limit_audio = int(model_cfg.get('limit_audio_per_prompt', 10))
    limit_image = int(model_cfg.get('limit_image_per_prompt', 1))
    limit_video = int(model_cfg.get('limit_video_per_prompt', 5))

    from vllm import AsyncLLMEngine, AsyncEngineArgs
    enable_prefix_cache = bool(model_cfg.get('enable_prefix_caching', False))
    max_new_mm = int(model_cfg.get('max_new_mm_per_request', 20))
    global _MAX_NEW_MM_PER_REQUEST
    _MAX_NEW_MM_PER_REQUEST = max_new_mm
    engine_args = AsyncEngineArgs(
        model=args.checkpoint_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=torch.cuda.device_count(),
        limit_mm_per_prompt={'image': limit_image, 'video': limit_video, 'audio': limit_audio},
        max_num_seqs=max_num_seqs,
        max_model_len=65535,
        seed=1234,
        enable_prefix_caching=enable_prefix_cache,
    )
    _logger.info(
        f"max_num_seqs={max_num_seqs}, "
        f"limit_mm_per_prompt={{image:{limit_image}, video:{limit_video}, audio:{limit_audio}}}, "
        f"enable_prefix_caching={enable_prefix_cache}, max_new_mm_per_request={max_new_mm}"
    )
    _logger.info(
        f"Model profile detected: {'instruct' if _MODEL_IS_INSTRUCT else 'thinking/other'} "
        f"(checkpoint={args.checkpoint_path})"
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
        _logger.info(
            f"Audio max duration overridden to {max_audio_sec}s "
            f"(n_samples={fe.n_samples}, nb_max_frames={fe.nb_max_frames})"
        )

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
    Convert a socket payload into (messages_list, temp_file_paths, total_input_audio_duration_sec).

    The payload uses the messages-based protocol where every turn is a
    structured message with inline base64 media.  This function
    materialises any base64 media blobs to temp files (so that
    process_mm_info / apply_chat_template can reference file paths),
    and returns a messages list ready for the model pipeline.

    Caller is responsible for deleting temp files afterwards.
    """
    messages = []
    temp_files = []
    total_input_audio_duration_sec = 0.0

    default_suffixes = {"audio": ".wav", "image": ".jpg", "video": ".mp4"}

    def _accumulate_audio_duration(item_obj: dict, item_type: str):
        nonlocal total_input_audio_duration_sec
        if item_type != "audio":
            return
        raw_ms = item_obj.get("duration_ms")
        if raw_ms is None:
            return
        try:
            total_input_audio_duration_sec += float(raw_ms) / 1000.0
        except Exception:
            pass

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
                out_item = {"type": item_type, item_type: path}
                # Copy through cache annotation fields if present
                for ann_key in ('item_id', 'duration_ms', 'sample_rate'):
                    if ann_key in item:
                        out_item[ann_key] = item[ann_key]
                out_content.append(out_item)
                _accumulate_audio_duration(item, item_type)

            elif item_type == "audio" and item.get("item_id") and not item.get("data"):
                # Stub reference — client believes this item is already cached.
                # Pass through annotation fields only; no temp file created.
                out_item = {"type": "audio", "item_id": item["item_id"]}
                for ann_key in ('duration_ms', 'sample_rate'):
                    if ann_key in item:
                        out_item[ann_key] = item[ann_key]
                out_content.append(out_item)
                _accumulate_audio_duration(item, item_type)

            elif item_type in default_suffixes and item.get(item_type):
                # Already a file path (e.g. local testing) — pass through
                out_content.append(item)
                _accumulate_audio_duration(item, item_type)

            elif item_type == "text":
                out_content.append({"type": "text", "text": item.get("text", "")})

            else:
                # Unknown item type — pass through
                out_content.append(item)

        if out_content:
            messages.append({"role": role, "content": out_content})

    return messages, temp_files, total_input_audio_duration_sec


def _count_text_tokens(processor, text: str) -> int:
    """Count text tokens with the model tokenizer."""
    if not text:
        return 0
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        # Some tokenizers may not accept add_special_tokens.
        return len(tokenizer.encode(text))
    except Exception:
        return 0


def _count_generated_tokens(processor, generated_text: str, request_output_item) -> int:
    """Count generated output tokens.

    Prefer native token IDs from vLLM outputs when available, and fall back
    to tokenizer-based counting of the current generated text.
    """
    token_ids = getattr(request_output_item, "token_ids", None)
    if isinstance(token_ids, (list, tuple)):
        return len(token_ids)
    return _count_text_tokens(processor, generated_text)


# ---------------------------------------------------------------------------
# Cache-aware multimodal encoding
# ---------------------------------------------------------------------------

_MM_MODALITIES = {"audio", "image", "video"}


def _encode_single_mm_item(item_type: str, file_path: str):
    """Run ``process_mm_info`` for a single media item and return the result.

    Wraps the item in a synthetic one-message list so that the existing
    ``process_mm_info`` machinery can process it.  Returns the first element
    of the corresponding modality list (audio / image / video), or ``None``
    if processing yields nothing.
    """
    synthetic_msg = [{"role": "user", "content": [{"type": item_type, item_type: file_path}]}]
    audios, images, videos = process_mm_info(synthetic_msg, use_audio_in_video=True)
    result_map = {"audio": audios, "image": images, "video": videos}
    result_list = result_map.get(item_type)
    if result_list is not None and len(result_list) > 0:
        return result_list[0]
    return None


def _process_mm_info_cached(
    messages: List[dict],
    session_cache: Optional[MmItemCache],
    request_id: str = '?',
) -> Tuple[Optional[list], Optional[list], Optional[list], int, List[dict]]:
    """Cache-aware replacement for ``process_mm_info``.

    Iterates content items in left-to-right message order (preserving the
    placeholder sequence that ``apply_chat_template`` expects).  For each
    audio / image / video item:

    * **Stub reference** (``item_id`` present but no file path): look up
      directly in *session_cache*; raise ``ValueError`` on miss.
    * **Full item** with ``item_id`` and *session_cache*: use
      ``get_or_encode`` for cache-or-encode.
    * Otherwise fall back to ``_encode_single_mm_item``.

    Returns ``(audios, images, videos, n_new, confirmed_items)`` where
    *n_new* counts fresh encodings and *confirmed_items* lists
    ``{item_id, meta_key}`` dicts for every item successfully resolved
    from (or stored into) the cache.
    """
    audios: List[Any] = []
    images: List[Any] = []
    videos: List[Any] = []
    n_new = 0
    n_hit = 0
    n_stub_hit = 0
    confirmed_items: List[dict] = []

    for msg in messages:
        for item in msg.get("content") or []:
            item_type = item.get("type")
            if item_type not in _MM_MODALITIES:
                continue

            item_id = item.get("item_id")
            meta_key = MmItemCache.meta_key_for(item) if item_id else None
            file_path = item.get(item_type)

            # --- Stub reference: no file_path, client expects cache hit ---
            if not file_path and item_id is not None and session_cache is not None:
                encoded = session_cache.get(item_id, meta_key)
                if encoded is None:
                    _logger.warning(
                        f"[{request_id}][STUB MISS] {item_type} item_id={item_id} "
                        f"meta={meta_key} — client desynced"
                    )
                    raise ValueError(
                        f"Stub cache miss for {item_type} item_id={item_id} "
                        f"meta={meta_key} — item not in server cache"
                    )
                n_stub_hit += 1
                _logger.debug(
                    f"[{request_id}][CACHE HIT stub] {item_type} item_id={item_id} "
                    f"meta={meta_key}"
                )
                confirmed_items.append({'item_id': item_id, 'meta_key': meta_key})
                {"audio": audios, "image": images, "video": videos}[item_type].append(encoded)
                continue

            # --- Full item with file_path ---
            if not file_path:
                continue

            encoded = None
            is_new = True

            if session_cache is not None and item_id is not None:
                encoded, is_new = session_cache.get_or_encode(
                    item_id, meta_key,
                    lambda fp=file_path, it=item_type: _encode_single_mm_item(it, fp),
                )
            else:
                encoded = _encode_single_mm_item(item_type, file_path)

            if encoded is None:
                _logger.warning(
                    f"[{request_id}] Encoding returned None for {item_type} "
                    f"item_id={item_id}"
                )
                continue

            if is_new:
                n_new += 1
                _logger.debug(
                    f"[{request_id}][CACHE MISS] {item_type} item_id={item_id} "
                    f"meta={meta_key} — encoded fresh"
                )
            else:
                n_hit += 1
                _logger.debug(
                    f"[{request_id}][CACHE HIT] {item_type} item_id={item_id} "
                    f"meta={meta_key}"
                )

            # Track confirmed cache entry (both newly stored and existing hits)
            if item_id is not None and meta_key is not None:
                confirmed_items.append({'item_id': item_id, 'meta_key': meta_key})

            {"audio": audios, "image": images, "video": videos}[item_type].append(encoded)

    _logger.info(
        f"[{request_id}] MM encoding: {n_stub_hit} stub hits, {n_hit} full hits, "
        f"{n_new} new encodings "
        f"(audio={len(audios)}, image={len(images)}, video={len(videos)})"
    )

    return (
        audios if audios else None,
        images if images else None,
        videos if videos else None,
        n_new,
        confirmed_items,
    )


# ---------------------------------------------------------------------------
# Preprocessing (synchronous — run in a thread to keep the event loop free)
# ---------------------------------------------------------------------------

def _prepare_inputs(processor, payload, session_cache: Optional[MmItemCache] = None):
    """Build vLLM inputs from a Socket.IO payload.

    This function is intentionally synchronous: apply_chat_template,
    process_mm_info, and base64-decoding are all CPU-bound and would block
    the asyncio event loop.  Callers should schedule it via
    ``asyncio.to_thread``.

    Returns:
      (inputs, sampling_params, temp_files, confirmed_items,
       input_text_tokens, input_audio_duration_sec)
    """
    from vllm import SamplingParams

    p = payload.get("params") or {}
    temperature   = float(p.get("temperature",  0.7))
    top_p         = float(p.get("top_p",        0.95))
    top_k         = int  (p.get("top_k",        20))
    max_tokens    = int  (p.get("max_tokens",   16384))
    thinking_mode = bool (p.get("thinking_mode", True))

    request_id = payload.get('request_id', '?')
    _logger.info(
        f"[{request_id}] Preparing inputs: thinking_mode={thinking_mode}, "
        f"max_tokens={max_tokens}, temperature={temperature}"
    )

    # ---------------------------------------------------------------------------
    # Thinking-mode control via the official Jinja2 chat template
    # ---------------------------------------------------------------------------
    # ``thinking_mode`` maps directly to ``enable_thinking`` in
    # ``processor.apply_chat_template``, which resolves to the model's own
    # ``chat_template.json``.  No <think>/<think> tokens are manually injected
    # here — the template handles everything:
    #
    #   enable_thinking=True  (default):
    #       Template emits: <|im_start|>assistant\n
    #       The model generates a <think>...</think> block naturally from
    #       training before producing its response.
    #
    #   enable_thinking=False:
    #       Template emits: <|im_start|>assistant\n<think>\n\n</think>\n\n
    #       The already-closed empty think block acts as a prefill — the model
    #       skips reasoning entirely and produces only the response.
    #
    # Guard: instruct model variants do not have the enable_thinking=False
    # template path (they were not trained with the closed think-block prefix).
    # _MODEL_IS_INSTRUCT overrides thinking_mode=False → True for those models.
    # ---------------------------------------------------------------------------
    template_enable_thinking = thinking_mode
    if _MODEL_IS_INSTRUCT and not thinking_mode:
        _logger.warning(
            f"[{request_id}] enable thinking is not active for instruct model; "
            f"ignoring thinking_mode=False and proceeding without think-tag suppression"
        )
        template_enable_thinking = True

    messages, temp_files, input_audio_duration_sec = _build_messages(payload)
    _logger.info(
        f"[{request_id}] _build_messages parsed {len(messages)} messages, "
        f"{len(temp_files)} temp files"
    )

    audios, images, videos, n_new, confirmed = _process_mm_info_cached(
        messages, session_cache, request_id,
    )
    _logger.info(
        f"[{request_id}] _process_mm_info_cached: "
        f"audios={len(audios) if audios else 0}, "
        f"images={len(images) if images else 0}, "
        f"videos={len(videos) if videos else 0}, "
        f"n_new={n_new}, confirmed={len(confirmed)}"
    )
    if n_new > _MAX_NEW_MM_PER_REQUEST:
        raise ValueError(
            f"Too many new multimodal items to encode in a single request: "
            f"{n_new} > limit {_MAX_NEW_MM_PER_REQUEST}. "
            f"Consider reducing context window size."
        )

    # -----------------------------------------------------------------------
    # Delta-thinking prefix support
    # -----------------------------------------------------------------------
    # When the client sends a non-empty ``thinking_prefix`` (raw text, no
    # ``<think>`` tag), and the model is a thinking-capable model with
    # thinking enabled, we:
    #   1. Append a partial assistant message whose content opens a ``<think>``
    #      block containing the prefix text.
    #   2. Use ``continue_final_message=True`` + ``add_generation_prompt=False``
    #      so the model continues generating inside the already-open think block.
    #
    # Guard: delta-thinking only activates when ALL of:
    #   - thinking_prefix is non-empty
    #   - thinking_mode is True (caller requested thinking)
    #   - model is NOT instruct-only (instruct models lack think-tag support)
    # -----------------------------------------------------------------------
    thinking_prefix = str(p.get("thinking_prefix", "") or "")
    _use_delta_thinking = bool(thinking_prefix and thinking_mode and not _MODEL_IS_INSTRUCT)

    if _use_delta_thinking:
        # Build the assistant message content: text prefix followed by any audio items.
        thinking_prefix_audio_items = p.get("thinking_prefix_audio_items") or []
        assistant_content = [{"type": "text", "text": f"<think>{thinking_prefix}"}]
        if thinking_prefix_audio_items:
            # Materialise base64 audio blobs to temp files so they pass through _build_messages.
            for audio_item in thinking_prefix_audio_items:
                if isinstance(audio_item, dict) and audio_item.get("type") == "audio" and audio_item.get("data"):
                    suffix = audio_item.get("suffix") or ".wav"
                    path = _save_temp_b64(audio_item["data"], suffix)
                    assistant_content.append({"type": "audio", "audio": path})
        messages.append({
            "role": "assistant",
            "content": assistant_content,
        })
        _logger.info(
            f"[{request_id}] Delta-thinking prefix injected "
            f"({len(thinking_prefix)} chars, {len(thinking_prefix_audio_items)} audio item(s)); "
            f"using continue_final_message=True"
        )

    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not _use_delta_thinking,
        **(dict(continue_final_message=True) if _use_delta_thinking else {}),
        enable_thinking=template_enable_thinking,
    )

    _logger.debug(
        f"[{request_id}] Final prompt ({len(prompt_text)} chars):\n{prompt_text}"
    )
    input_text_tokens = _count_text_tokens(processor, prompt_text)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|im_start|>"],
    )

    inputs = {
        "prompt": prompt_text,
        "multi_modal_data": {},
        "mm_processor_kwargs": {"use_audio_in_video": True},
    }
    if images is not None: inputs["multi_modal_data"]["image"] = images
    if videos is not None: inputs["multi_modal_data"]["video"] = videos
    if audios is not None: inputs["multi_modal_data"]["audio"] = audios

    return (
      inputs,
      sampling_params,
      temp_files,
      confirmed,
      input_text_tokens,
      input_audio_duration_sec,
    )


# ---------------------------------------------------------------------------
# Core streaming generation coroutine
# ---------------------------------------------------------------------------

async def _stream_generate(sio, sid, model, processor, payload,
                           session_cache: Optional[MmItemCache] = None):

    request_id = payload.get('request_id') or str(uuid.uuid4())
    p = payload.get('params') or {}
    thinking_mode = bool(p.get('thinking_mode', True))
    _logger.info(f"[{request_id}] Generation request from sid={sid}, thinking_mode={thinking_mode}")
    # _logger.debug(f"[{request_id}] Incoming payload: {payload}")
    
    open_tag = SERVER_CONFIG.get('thinking', {}).get('open_tag', '<think>')
    close_tag = SERVER_CONFIG.get('thinking', {}).get('close_tag', '</think>')

    # Offload blocking preprocessing (base64 decode, tokenization,
    # audio/image feature extraction) to a worker thread so the event
    # loop stays responsive for new connections and other events.
    (
      inputs,
      sampling_params,
      temp_files,
      confirmed_items,
      input_text_tokens,
      input_audio_duration_sec,
    ) = await asyncio.to_thread(
        _prepare_inputs, processor, payload, session_cache,
    )

    # Notify client which items are confirmed in the encoding cache.
    # Emitted before generation_start so the client has acks before tokens.
    if confirmed_items:
        await sio.emit("items_cached", {
            "request_id": request_id,
            "items": confirmed_items,
        }, to=sid)

    await sio.emit("generation_start", {
      "request_id": request_id,
      "input_text_tokens": input_text_tokens,
      "input_audio_duration_sec": round(float(input_audio_duration_sec), 3),
    }, to=sid)

    t_start      = time.perf_counter()
    prev_text    = ""
    n_tokens     = 0
    ttft         = None
    _t_first_response_token = None
    _think_started = False
    _think_ended   = False
    _t_think_end   = None   # elapsed when </think> was first detected

    # Delta-thinking prefix: if the client sent a non-empty thinking_prefix
    # with thinking enabled on a thinking-capable model, generation starts
    # inside an already-open <think> block.
    _thinking_prefix = str(p.get('thinking_prefix', '') or '')
    if _thinking_prefix and thinking_mode and not _MODEL_IS_INSTRUCT:
        _think_started = True
        _logger.info(f"[{request_id}] Delta-thinking mode: _think_started pre-set to True")

    try:
        async for output in model.generate(inputs, sampling_params, request_id):
            output_item = output.outputs[0] if output.outputs else None
            full_text = output_item.text if output_item is not None else prev_text

            prev_n_tokens = n_tokens
            n_tokens = _count_generated_tokens(processor, full_text, output_item)
            elapsed = time.perf_counter() - t_start
            # TTFT is anchored to the first observed generated token.
            if ttft is None and n_tokens > prev_n_tokens:
                ttft = elapsed
                _logger.debug(f"[{request_id}] First token in {ttft:.3f}s")
            
            delta = full_text[len(prev_text):]
            if delta:
                # Detect thinking block transitions and log them
                if not _think_started and open_tag in full_text:
                    _think_started = True
                    _logger.info(f"[{request_id}] Thinking block STARTED — model is reasoning")
                if _think_started and not _think_ended and close_tag in full_text:
                    _think_ended = True
                    _t_think_end = elapsed
                    think_chars = full_text.find(close_tag) - full_text.find(open_tag) - len(open_tag)
                    _logger.info(
                        f"[{request_id}] Thinking block ENDED (~{think_chars} chars of reasoning)"
                    )

                # Track first response token (non-thinking content).
                # If no thinking block: set on first delta that is not a leading
                #   prefix of open_tag (defers while "<", "<t", "<th"… are
                #   assembling, so we don't lock in ttft before <think> is detected).
                # If thinking block present: set on first response char after close_tag.
                # Always use elapsed (current time) rather than the stale ttft value.
                if _t_first_response_token is None:
                    if _think_ended:
                        close_pos = full_text.find(close_tag)
                        response_content = full_text[close_pos + len(close_tag):].strip()
                        if response_content:
                            _t_first_response_token = elapsed
                    elif not _think_started:
                        # Defer while full_text is still a leading prefix of open_tag
                        # (e.g. "<", "<t", "<th"…) — the tag may still be assembling.
                        partial = full_text.lstrip()
                        if not (partial and open_tag.startswith(partial)):
                            _t_first_response_token = elapsed

                token_payload = {
                    "request_id": request_id,
                    "delta":      delta,
                    "full_text":  full_text,
                    "num_tokens": n_tokens,
                    "elapsed":    round(elapsed, 3),
                    "ttft":       round(ttft, 3),
                    "finished":   output.finished,
                    "input_text_tokens": input_text_tokens,
                    "input_audio_duration_sec": round(float(input_audio_duration_sec), 3),
                }

                # Attach terminal efficiency stats on the final token event
                # so clients that finalize on token.finished can still report
                # complete metrics even if generation_complete arrives later.
                if output.finished:
                  total_time = elapsed
                  post_ttft_duration = max(0.0, total_time - (ttft or 0.0))
                  tps = n_tokens / total_time if total_time > 0 else 0.0
                  _t_first_response_token_final = _t_first_response_token if _t_first_response_token is not None else ttft
                  token_payload.update({
                  "full_generation_latency_sec": round(total_time, 3),
                  "total_time": round(total_time, 3),  # legacy alias
                    "tokens_per_second": round(tps, 1),
                    # generation_duration is post-first-token generation time.
                    "generation_duration": round(post_ttft_duration, 3),
                    "generated_tokens": n_tokens,
                    "time_to_first_token": round(ttft, 3) if ttft is not None else None,
                    "llm_time_to_first_response_token": round(_t_first_response_token_final, 3) if _t_first_response_token_final is not None else None,
                    "thinking_time_to_first_token": round(ttft, 3) if (_think_started and ttft is not None) else 0.0,
                    "thinking_duration": round(max(0.0, _t_think_end - ttft), 3) if (_think_started and _think_ended and _t_think_end is not None and ttft is not None) else 0.0,
                    "input_text_tokens": input_text_tokens,
                    "input_audio_duration_sec": round(float(input_audio_duration_sec), 3),
                  })

                await sio.emit("token", token_payload, to=sid)
                prev_text = full_text

        # Fallback: if thinking never ended or no response content detected,
        # time_to_first_response_token equals ttft.
        if _t_first_response_token is None:
            _t_first_response_token = ttft

        total_time = time.perf_counter() - t_start
        post_ttft_duration = max(0.0, total_time - (ttft or 0.0))
        tps = n_tokens / total_time if total_time > 0 else 0.0
        if _think_started:
            _logger.info(
                f"[{request_id}] Generation complete: {n_tokens} tokens, {tps:.1f} tps, "
                f"thinking={'complete' if _think_ended else 'block did not close'}"
            )
        else:
            _logger.info(
                f"[{request_id}] Generation complete: {n_tokens} tokens, {tps:.1f} tps — "
                f"no {open_tag} block detected (thinking_mode=False or model skipped reasoning)"
            )
        await sio.emit("generation_complete", {
            "request_id":        request_id,
            "full_text":         prev_text,
          "full_generation_latency_sec": round(total_time, 3),
          "total_time":        round(total_time,  3),  # legacy alias
            "num_tokens":        n_tokens,
            "tokens_per_second": round(tps, 1),
            "ttft":              round(ttft, 3) if ttft is not None else None,
          # Explicit efficiency aliases for downstream consumers.
          # generation_duration is post-first-token generation time.
          "generation_duration": round(post_ttft_duration, 3),
          "generated_tokens":    n_tokens,
          "time_to_first_token": round(ttft, 3) if ttft is not None else None,
          "llm_time_to_first_response_token": round(_t_first_response_token, 3) if _t_first_response_token is not None else None,
          "thinking_time_to_first_token": round(ttft, 3) if (_think_started and ttft is not None) else 0.0,
          "thinking_duration": round(max(0.0, _t_think_end - ttft), 3) if (_think_started and _think_ended and _t_think_end is not None and ttft is not None) else 0.0,
          "input_text_tokens":   input_text_tokens,
          "input_audio_duration_sec": round(float(input_audio_duration_sec), 3),
        }, to=sid)

    except asyncio.CancelledError:
        _logger.info(f"[{request_id}] Generation cancelled (stopped by client)")
        total_time = time.perf_counter() - t_start
        post_ttft_duration = max(0.0, total_time - (ttft or 0.0))
        tps = n_tokens / total_time if total_time > 0 else 0.0
        await sio.emit("generation_stopped", {
            "request_id":  request_id,
            "partial_text": prev_text,
          "full_generation_latency_sec": round(total_time, 3),
          "total_time":        round(total_time, 3),  # legacy alias
            "num_tokens":        n_tokens,
            "tokens_per_second": round(tps, 1),
            "ttft":              round(ttft, 3) if ttft is not None else None,
            # Explicit efficiency aliases for downstream consumers.
          # generation_duration is post-first-token generation time.
          "generation_duration": round(post_ttft_duration, 3),
            "generated_tokens":    n_tokens,
            "time_to_first_token": round(ttft, 3) if ttft is not None else None,
            "llm_time_to_first_response_token": round(_t_first_response_token, 3) if _t_first_response_token is not None else None,
            "thinking_time_to_first_token": round(ttft, 3) if (_think_started and ttft is not None) else 0.0,
            "thinking_duration": round(max(0.0, _t_think_end - ttft), 3) if (_think_started and _think_ended and _t_think_end is not None and ttft is not None) else 0.0,
            "input_text_tokens":   input_text_tokens,
            "input_audio_duration_sec": round(float(input_audio_duration_sec), 3),
        }, to=sid)

    except Exception as exc:
        _logger.error(f"[{request_id}] Generation error: {exc}", exc_info=True)
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
        max_http_buffer_size=200 * 1024 * 1024,   # 200 MB – enough for audio/video
    )
    app = web.Application()
    sio.attach(app)

    _active = {}   # sid -> (asyncio.Task, request_id: str)
    _session_caches: Dict[str, MmItemCache] = {}  # sid -> per-session encoding cache
    _connected_sids: set = set()  # sids with live connections; used by clear_cache to find stale entries

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
                _logger.warning(f"model.abort({req_id}) failed ({reason}): {e}")
            task.cancel()
            _logger.info(f"{reason}: cancelled task for sid={sid} (request_id={req_id})")

    @sio.on("connect")
    async def on_connect(sid, environ):
        _logger.info(f"connect    sid={sid}")
        _connected_sids.add(sid)
        _session_caches[sid] = MmItemCache()
        await sio.emit("server_ready", {"sid": sid}, to=sid)

    @sio.on("disconnect")
    async def on_disconnect(sid):
        _logger.info(f"disconnect sid={sid}")
        _connected_sids.discard(sid)
        cache = _session_caches.pop(sid, None)
        if cache is not None:
            _logger.info(f"disconnect sid={sid}: released encoding cache ({len(cache)} entries)")
        await _abort_active(sid, "disconnect")

    @sio.on("generate")
    async def on_generate(sid, payload):
        # Cancel any in-flight generation before starting a new one
        await _abort_active(sid, "new generate")
        request_id = payload.get('request_id') or str(uuid.uuid4())
        # Inject the resolved request_id back so _stream_generate uses it
        payload['request_id'] = request_id
        task = asyncio.create_task(
            _stream_generate(sio, sid, model, processor, payload,
                             session_cache=_session_caches.get(sid))
        )
        _active[sid] = (task, request_id)

    @sio.on("stop")
    async def on_stop(sid, payload=None):
        await _abort_active(sid, "stop")

    @sio.on("clear_cache")
    async def on_clear_cache(sid):
        """Wipe cached state for all sessions that are no longer connected.

        Safe to call from any live client.  Aborts any in-flight vLLM
        requests belonging to stale sessions, releases their MmItemCache
        entries, then emits a bare ``cache_cleared`` ack to the caller.
        """
        stale_sids = (set(_session_caches.keys()) | set(_active.keys())) - _connected_sids
        for stale_sid in stale_sids:
            cache = _session_caches.pop(stale_sid, None)
            await _abort_active(stale_sid, "clear_cache")
            _logger.info(
                f"clear_cache: released sid={stale_sid} "
                f"(cache entries: {len(cache) if cache is not None else 0})"
            )
        _logger.info(
            f"clear_cache: swept {len(stale_sids)} stale session(s), triggered by sid={sid}"
        )
        await sio.emit("cache_cleared", to=sid)

    @sio.on("reset_kv_cache")
    async def on_reset_kv_cache(sid):
        """Flush vLLM's prefix KV cache.

        Calls AsyncLLMEngine.reset_prefix_cache(), which discards all cached
        KV blocks so the next request runs with a cold cache.  Intended for
        evaluation harnesses that need fair, order-independent TTFT comparisons.
        Emits ``kv_cache_reset`` ack to the caller when complete.
        """
        _logger.info(f"reset_kv_cache: flushing prefix KV cache (triggered by sid={sid})")
        try:
            await model.reset_prefix_cache()
            _logger.info(f"reset_kv_cache: prefix KV cache cleared (triggered by sid={sid})")
        except Exception as e:
            _logger.warning(f"reset_kv_cache: reset_prefix_cache() failed: {e}")
        await sio.emit("kv_cache_reset", to=sid)

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
      <span>Full Latency&nbsp;<span class="mv" id="m-total">—</span></span>
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
    const fullLatency = d.full_generation_latency_sec ?? d.total_time;
    setMetrics(
      d.ttft != null ? d.ttft + 's' : '—',
      d.num_tokens,
      d.tokens_per_second + ' t/s',
      fullLatency != null ? (fullLatency + 's') : '—'
    );
    setStatus('done');
    history.push({ role: 'assistant', content: [{type: 'text', text: d.full_text}] });
    setGenerating(false);
  });

  socket.on('generation_stopped', (d) => {
    finalizeAsst(d.partial_text || null);
    if (d.num_tokens != null || d.generated_tokens != null) {
      const nTok = d.generated_tokens ?? d.num_tokens;
      const fullLatency = d.full_generation_latency_sec ?? d.total_time;
      const ttft = d.time_to_first_token ?? d.ttft;
      const tps = d.tokens_per_second;
      setMetrics(
        ttft != null ? ttft + 's' : '—',
        nTok ?? '—',
        tps != null ? (tps + ' t/s') : '—',
        fullLatency != null ? (fullLatency + 's') : '—'
      );
    }
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

    # Set up the file-backed logger (writes to logger/logs/socketio_server/)
    _logger_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logger')
    sys.path.insert(0, os.path.dirname(_logger_dir))
    from logger.logger import setup_logger as _setup_logger
    _logger = _setup_logger('socketio_server', file_log_level='DEBUG', terminal_log_level='INFO')
    sys.path.pop(0)

    _logger.info(f"Loading model from: {args.checkpoint_path}")
    model, processor = _load_model_processor(args)
    _logger.info(f"Model loaded. Serving GUI + Socket.IO on http://{args.host}:{args.port}")
    app = create_socketio_app(model, processor)
    # Suppress the default "Running on ..." banner from aiohttp
    web.run_app(app, host=args.host, port=args.port, print=lambda _: None)
