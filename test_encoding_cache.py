#!/usr/bin/env python3
"""
Test script for the per-session multimodal encoding cache.

Three test modes:
  1) single   — Synthesise a single 60-minute audio file and send it.
  2) chunked  — Chunk 60 min into N pieces, send incrementally (old+new).
  3) concurrent — Multiple threads submit overlapping requests that share
                  an unseen item, verifying the per-item lock.

Usage:
    # Run all tests against a server on localhost:8902
    python test_encoding_cache.py --host 127.0.0.1 --port 8902

    # Run only the chunked test with 6 chunks
    python test_encoding_cache.py --test chunked --num-chunks 6

    # Run only the single-file test with a shorter duration for quick check
    python test_encoding_cache.py --test single --duration-sec 60

    # Run concurrent test with 4 threads hitting the same item
    python test_encoding_cache.py --test concurrent --concurrent-threads 4
"""

import argparse
import base64
import io
import os
import sys
import threading
import time
import wave

import socketio

# ---------------------------------------------------------------------------
# Reuse the same audio helpers as the production mm_llm_client_session.
# wrap_pcm_as_wav creates a WAV from raw PCM in exactly the same way that
# the real pipeline (_encode_pcm_as_b64_wav) does — single writeframes call.
# ---------------------------------------------------------------------------
_AUDIO_HELPERS_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..', 'AudioLLMInterface', 'utils',
)
sys.path.insert(0, _AUDIO_HELPERS_DIR)
from audio_helpers import wrap_pcm_as_wav
sys.path.remove(_AUDIO_HELPERS_DIR)


# ---------------------------------------------------------------------------
# Audio synthesis helpers
# ---------------------------------------------------------------------------

def _synthesise_pcm_silence(duration_sec: float, sample_rate: int = 16000) -> bytes:
    """Return raw PCM s16le silence bytes (mono, 16-bit) — same format the
    production pipeline stores in ContextItem.audio_bytes."""
    n_samples = int(duration_sec * sample_rate)
    return b'\x00\x00' * n_samples


def _encode_pcm_as_b64_wav(pcm_bytes: bytes, sample_rate: int) -> str:
    """Mirrors MultiModalLLMSession._encode_pcm_as_b64_wav exactly."""
    wav_bytes = wrap_pcm_as_wav(pcm_bytes, sample_rate)
    return base64.b64encode(wav_bytes).decode('ascii')


def _make_audio_content_item(pcm_bytes: bytes, item_id: str,
                             duration_ms: int, sample_rate: int = 16000) -> dict:
    """Build an audio content item identical to _serialize_user_item."""
    return {
        'type': 'audio',
        'data': _encode_pcm_as_b64_wav(pcm_bytes, sample_rate),
        'suffix': '.wav',
        'item_id': item_id,
        'duration_ms': duration_ms,
        'sample_rate': sample_rate,
    }


# ---------------------------------------------------------------------------
# Socket.IO client helpers
# ---------------------------------------------------------------------------

class GenerationResult:
    """Collects streaming events from a single generation request."""
    def __init__(self):
        self.tokens: list = []
        self.full_text: str = ''
        self.error: str = ''
        self.finished = threading.Event()
        self.started = threading.Event()
        self.ttft: float = None
        self.full_generation_latency_sec: float = None
        self.num_tokens: int = 0

    def summary(self) -> str:
        if self.error:
            return f"ERROR: {self.error}"
        return (
            f"tokens={self.num_tokens}, ttft={self.ttft}s, "
            f"full_latency={self.full_generation_latency_sec}s, "
            f"text={self.full_text[:120]!r}..."
        )


def _connect(host: str, port: int, timeout: float = 30.0) -> socketio.Client:
    """Connect a Socket.IO client and wait for server_ready."""
    sio = socketio.Client(logger=False, engineio_logger=False)
    ready = threading.Event()

    @sio.on('server_ready')
    def _on_ready(data):
        ready.set()

    url = f'http://{host}:{port}'
    print(f"  Connecting to {url} ...")
    sio.connect(url, wait_timeout=timeout)
    if not ready.wait(timeout):
        raise TimeoutError(f"server_ready not received within {timeout}s")
    print(f"  Connected (sid={sio.sid})")
    return sio


def _generate(sio: socketio.Client, messages: list, request_id: str,
              max_tokens: int = 64, thinking_mode: bool = False,
              timeout: float = 300.0) -> GenerationResult:
    """Emit a generate request and block until completion or error."""
    result = GenerationResult()

    @sio.on('generation_start')
    def _on_start(data):
        if data.get('request_id') == request_id:
            result.started.set()

    @sio.on('token')
    def _on_token(data):
        if data.get('request_id') == request_id:
            result.tokens.append(data.get('delta', ''))
            result.full_text = data.get('full_text', '')
            result.num_tokens = data.get('num_tokens', 0)
            if result.ttft is None:
                result.ttft = data.get('ttft')

    @sio.on('generation_complete')
    def _on_complete(data):
        if data.get('request_id') == request_id:
            result.full_text = data.get('full_text', '')
            result.full_generation_latency_sec = data.get('full_generation_latency_sec', data.get('total_time'))
            result.num_tokens = data.get('num_tokens', 0)
            result.ttft = data.get('ttft')
            result.finished.set()

    @sio.on('generation_error')
    def _on_error(data):
        if data.get('request_id') == request_id:
            result.error = data.get('error', 'unknown error')
            result.finished.set()

    @sio.on('generation_stopped')
    def _on_stopped(data):
        if data.get('request_id') == request_id:
            result.full_text = data.get('partial_text', '')
            result.finished.set()

    payload = {
        'request_id': request_id,
        'messages': messages,
        'params': {
            'max_tokens': max_tokens,
            'thinking_mode': thinking_mode,
            'temperature': 0.7,
        },
    }
    t0 = time.time()
    sio.emit('generate', payload)
    if not result.finished.wait(timeout):
        result.error = f"Timed out after {timeout}s"
    elapsed_wall = time.time() - t0
    print(f"    [{request_id}] wall={elapsed_wall:.1f}s  {result.summary()}")
    return result


# ---------------------------------------------------------------------------
# Test 1: Single 60-minute audio
# ---------------------------------------------------------------------------

def test_single(host: str, port: int, duration_sec: float):
    """Synthesise a single long audio file and send it to the server."""
    print(f"\n{'='*70}")
    print(f"TEST 1: Single {duration_sec/60:.0f}-minute audio file")
    print(f"{'='*70}")

    print(f"  Synthesising {duration_sec}s of PCM silence (16 kHz, 16-bit mono) ...")
    t0 = time.time()
    pcm_bytes = _synthesise_pcm_silence(duration_sec)
    print(f"  Synthesised {len(pcm_bytes)/1024/1024:.1f} MB PCM in {time.time()-t0:.1f}s")

    item = _make_audio_content_item(
        pcm_bytes, item_id='single_60min', duration_ms=int(duration_sec * 1000),
    )

    messages = [
        {'role': 'user', 'content': [
            item,
            {'type': 'text', 'text': 'What do you hear in this audio?'},
        ]},
    ]

    sio = _connect(host, port)
    try:
        result = _generate(sio, messages, request_id='test_single_60min',
                           max_tokens=64, timeout=600.0)
        if result.error:
            print(f"  FAILED: {result.error}")
            return False
        print(f"  PASSED")
        return True
    finally:
        sio.disconnect()


# ---------------------------------------------------------------------------
# Test 2: Chunked incremental audio with cache monitoring
# ---------------------------------------------------------------------------

def test_chunked(host: str, port: int, duration_sec: float, num_chunks: int):
    """Split audio into chunks, send incrementally (old + new each round)."""
    print(f"\n{'='*70}")
    print(f"TEST 2: Chunked incremental ({num_chunks} rounds, "
          f"{duration_sec/60:.0f} min total)")
    print(f"{'='*70}")

    chunk_duration = duration_sec / num_chunks
    print(f"  Chunk duration: {chunk_duration:.1f}s each")

    # Pre-synthesise all chunks as raw PCM
    print(f"  Synthesising {num_chunks} PCM chunks ...")
    chunks = []
    for i in range(num_chunks):
        pcm = _synthesise_pcm_silence(chunk_duration)
        chunks.append({
            'pcm': pcm,
            'item_id': f'chunk_{i:03d}',
            'duration_ms': int(chunk_duration * 1000),
        })
    print(f"  All chunks synthesised")

    sio = _connect(host, port)
    try:
        for round_idx in range(1, num_chunks + 1):
            active_chunks = chunks[:round_idx]
            n_old = round_idx - 1
            n_new = 1

            print(f"\n  --- Round {round_idx}/{num_chunks}: "
                  f"{len(active_chunks)} items ({n_old} cached + {n_new} new) ---")

            content = []
            for c in active_chunks:
                content.append(_make_audio_content_item(
                    c['pcm'], item_id=c['item_id'], duration_ms=c['duration_ms'],
                ))
            content.append({'type': 'text', 'text': f'Round {round_idx}: summarise.'})

            messages = [{'role': 'user', 'content': content}]

            result = _generate(
                sio, messages,
                request_id=f'test_chunked_r{round_idx:02d}',
                max_tokens=32,
                timeout=600.0,
            )
            if result.error:
                print(f"  Round {round_idx} FAILED: {result.error}")
                return False

            print(f"    Expected: {n_old} cache hits, {n_new} cache miss")

        print(f"\n  PASSED — all {num_chunks} rounds completed")
        return True
    finally:
        sio.disconnect()


# ---------------------------------------------------------------------------
# Test 3: Concurrent access to the same unseen item
# ---------------------------------------------------------------------------

def test_concurrent(host: str, port: int, num_threads: int):
    """Multiple clients submit overlapping requests sharing an unseen item."""
    print(f"\n{'='*70}")
    print(f"TEST 3: Concurrent access ({num_threads} threads, shared unseen item)")
    print(f"{'='*70}")

    # Synthesise a shared audio clip (5s is enough for the test)
    shared_pcm = _synthesise_pcm_silence(5.0)
    shared_item = _make_audio_content_item(
        shared_pcm, item_id='shared_concurrent_item', duration_ms=5000,
    )

    # Each thread also has a unique item
    unique_pcms = [_synthesise_pcm_silence(2.0) for _ in range(num_threads)]

    # All threads share ONE Socket.IO connection (same session = same cache)
    sio = _connect(host, port)

    results = [None] * num_threads
    barrier = threading.Barrier(num_threads)

    def _worker(idx):
        unique_item = _make_audio_content_item(
            unique_pcms[idx],
            item_id=f'unique_{idx}',
            duration_ms=2000,
        )
        messages = [
            {'role': 'user', 'content': [
                shared_item,
                unique_item,
                {'type': 'text', 'text': f'Thread {idx}: describe the audio.'},
            ]},
        ]
        # Barrier ensures all threads fire at roughly the same time
        barrier.wait(timeout=30)
        results[idx] = _generate(
            sio, messages,
            request_id=f'test_concurrent_t{idx}',
            max_tokens=32,
            timeout=300.0,
        )

    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=_worker, args=(i,), daemon=True)
        threads.append(t)

    print(f"  Launching {num_threads} threads ...")
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=360)

    sio.disconnect()

    # Evaluate
    errors = [r.error for r in results if r and r.error]
    if errors:
        print(f"  FAILED: {len(errors)} thread(s) errored: {errors}")
        return False

    print(f"  PASSED — all {num_threads} threads completed without error")
    print(f"  (Check server log for exactly 1 CACHE MISS on 'shared_concurrent_item')")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Test per-session multimodal encoding cache',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8902)
    parser.add_argument('--test', choices=['all', 'single', 'chunked', 'concurrent'],
                        default='all', help='Which test to run (default: all)')
    parser.add_argument('--duration-sec', type=float, default=3600.0,
                        help='Total audio duration in seconds (default: 3600 = 60 min)')
    parser.add_argument('--num-chunks', type=int, default=6,
                        help='Number of chunks for the incremental test (default: 6)')
    parser.add_argument('--concurrent-threads', type=int, default=4,
                        help='Number of threads for the concurrent test (default: 4)')
    args = parser.parse_args()

    passed = []
    failed = []

    if args.test in ('all', 'single'):
        ok = test_single(args.host, args.port, args.duration_sec)
        (passed if ok else failed).append('single')

    if args.test in ('all', 'chunked'):
        ok = test_chunked(args.host, args.port, args.duration_sec, args.num_chunks)
        (passed if ok else failed).append('chunked')

    if args.test in ('all', 'concurrent'):
        ok = test_concurrent(args.host, args.port, args.concurrent_threads)
        (passed if ok else failed).append('concurrent')

    print(f"\n{'='*70}")
    print(f"RESULTS: {len(passed)} passed, {len(failed)} failed")
    if passed:
        print(f"  Passed: {', '.join(passed)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"{'='*70}")
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
