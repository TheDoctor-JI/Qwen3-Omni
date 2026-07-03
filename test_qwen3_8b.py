#!/usr/bin/env python3
"""
Integration tests for Qwen3-8B Socket.IO server.

Usage:
    python test_qwen3_8b.py [--host HOST] [--port PORT]

Tests:
    1. Basic generation (thinking ON)
    2. Basic generation (thinking OFF)
    3. Thinking prefix (delta-thinking)
    4. Max thinking tokens → burnout + synthetic close tag
    5. Stop mid-generation
    6. Reset KV cache
    7. Efficiency metrics (all fields present)

Requires:
    pip install "python-socketio[client]"
"""

import argparse
import os
import sys
import time

import socketio

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Integration tests for Qwen3-8B Socket.IO server")
    p.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8902, help="Server port (default: 8902)")
    p.add_argument("--timeout", type=int, default=120, help="Per-test timeout in seconds (default: 120)")
    p.add_argument("--test", type=str, nargs="*", default=None,
                   help="Specific tests to run (e.g., 'thinking_on' 'stop'). Default: run all.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Test fixture
# ---------------------------------------------------------------------------
class TestFixture:
    """Connects to a running Socket.IO server and provides helpers."""

    def __init__(self, host: str, port: int, timeout: float = 120.0):
        self._url = f"http://{host}:{port}"
        self._sio = socketio.Client()
        self._timeout = timeout
        self._connected = False
        self._last_event: dict = {}
        self._events: list = []  # all events received, in order
        self._tokens: list = []
        self._server_ready = False

        # Register callbacks
        @self._sio.on("connect")
        def _on_connect():
            self._connected = True

        @self._sio.on("disconnect")
        def _on_disconnect():
            self._connected = False

        @self._sio.on("*")
        def _on_any(event, data=None):
            record = {"event": event, "data": data, "time": time.time()}
            self._events.append(record)
            self._last_event = record
            if event == "token":
                self._tokens.append(data)

        # Blocking connect
        self._sio.connect(self._url, transports=["websocket"])
        deadline = time.time() + timeout
        while not self._connected:
            if time.time() > deadline:
                raise TimeoutError(f"Failed to connect to {self._url} within {timeout}s")
            time.sleep(0.1)

        # Wait for server_ready
        deadline = time.time() + timeout
        self._server_ready = False
        while not self._server_ready:
            if time.time() > deadline:
                raise TimeoutError("server_ready not received")
            for e in self._events:
                if e["event"] == "server_ready":
                    self._server_ready = True
                    break
            time.sleep(0.1)

        # Clear events after connect
        self._events.clear()
        self._tokens.clear()
        self._last_event = {}

    def close(self):
        if self._sio.connected:
            self._sio.disconnect()

    def emit(self, event: str, data: dict = None):
        self._sio.emit(event, data or {})

    def wait_for_event(self, event_name: str, timeout: float = None) -> dict:
        """Block until a specific event is received, return its data."""
        deadline = time.time() + (timeout or self._timeout)
        while True:
            for i, e in enumerate(self._events):
                if e["event"] == event_name:
                    return e["data"]
            if time.time() > deadline:
                raise TimeoutError(f"Event '{event_name}' not received within timeout")
            time.sleep(0.1)

    def wait_for_any(self, event_names: list, timeout: float = None) -> tuple:
        """Block until any of the given events is received, return (name, data)."""
        deadline = time.time() + (timeout or self._timeout)
        while True:
            for e in self._events:
                if e["event"] in event_names:
                    return (e["event"], e["data"])
            if time.time() > deadline:
                raise TimeoutError(f"None of {event_names} received within timeout")
            time.sleep(0.1)

    def wait_for_token_count(self, min_count: int, timeout: float = None) -> list:
        """Wait until at least min_count token events have been received."""
        deadline = time.time() + (timeout or self._timeout)
        while len(self._tokens) < min_count:
            if time.time() > deadline:
                raise TimeoutError(f"Only {len(self._tokens)} tokens received, expected >= {min_count}")
            time.sleep(0.1)
        return list(self._tokens)

    def clear_events(self):
        self._events.clear()
        self._tokens.clear()
        self._last_event = {}

    def request_id(self):
        return f"test_{os.urandom(4).hex()}"


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------
_PASS = 0
_FAIL = 0

def ok(name: str, condition: bool, detail: str = ""):
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  ✅ {name}")
    else:
        _FAIL += 1
        print(f"  ❌ {name}  — {detail}" if detail else f"  ❌ {name}")


def section(name: str):
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_thinking_on(fix: TestFixture):
    section("Test 1: Thinking mode ON")
    fix.clear_events()
    rid = fix.request_id()
    fix.emit("generate", {
        "request_id": rid,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "What is 2 + 2?"}]}],
        "params": {"thinking_mode": True, "max_tokens": 512},
    })

    # Wait for generation_start
    gen_start = fix.wait_for_event("generation_start")
    ok("generation_start received", gen_start.get("request_id") == rid)
    ok("input_text_tokens > 0", gen_start.get("input_text_tokens", 0) > 0)
    ok("input_audio_duration_sec is 0.0", gen_start.get("input_audio_duration_sec") == 0.0)

    # Wait for completion
    comp = fix.wait_for_event("generation_complete")
    full_text = comp.get("full_text", "")
    ok("generation_complete received", True)
    ok("full_text contains <think>", "<think>" in full_text, f"full_text[:100]={full_text[:100]!r}")
    ok("full_text contains </think>", "</think>" in full_text)

    # Efficiency metrics
    ok("num_tokens > 0", comp.get("num_tokens", 0) > 0)
    ok("ttft is not None", comp.get("ttft") is not None)
    ok("tokens_per_second > 0", comp.get("tokens_per_second", 0) > 0)
    ok("full_generation_latency_sec > 0", comp.get("full_generation_latency_sec", 0) > 0)

    print(f"  [info] tokens={comp.get('num_tokens')}, ttft={comp.get('ttft')}s, tps={comp.get('tokens_per_second')}")


def test_thinking_off(fix: TestFixture):
    section("Test 2: Thinking mode OFF")
    fix.clear_events()
    rid = fix.request_id()
    fix.emit("generate", {
        "request_id": rid,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello in exactly one word."}]}],
        "params": {"thinking_mode": False, "max_tokens": 128},
    })

    comp = fix.wait_for_event("generation_complete")
    full_text = comp.get("full_text", "")
    ok("generation_complete received", True)
    ok("full_text does NOT contain <think>", "<think>" not in full_text,
       f"full_text={full_text!r}")
    ok("full_text is non-empty", len(full_text.strip()) > 0)
    ok("ttft is not None", comp.get("ttft") is not None)


def test_thinking_prefix(fix: TestFixture):
    section("Test 3: Thinking prefix (delta-thinking)")
    fix.clear_events()
    rid = fix.request_id()
    prefix = "First, note that 3 + 5 = 8."
    fix.emit("generate", {
        "request_id": rid,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "What is 6 + 7?"}]}],
        "params": {"thinking_mode": True, "max_tokens": 512, "thinking_prefix": prefix},
    })

    tokens = fix.wait_for_token_count(min_count=2)
    comp = fix.wait_for_event("generation_complete")
    full_text = comp.get("full_text", "")

    ok("generation_complete received", True)
    # The thinking prefix is injected into the PROMPT (not generated),
    # so full_text only contains what the model generates after the prefix.
    # The model should close the think block: check for </think>.
    ok("full_text contains </think> (model closed the think block)",
       "</think>" in full_text,
       f"full_text[:200]={full_text[:200]!r}")
    ok("full_text has content beyond think block",
       len(full_text.split("</think>", 1)[-1].strip()) > 0 if "</think>" in full_text else False)
    ok("at least some tokens received", len(tokens) >= 1)


def test_max_thinking_tokens_burnout(fix: TestFixture):
    section("Test 4: max_thinking_tokens → burnout + synthetic close tag")
    fix.clear_events()
    rid = fix.request_id()
    fix.emit("generate", {
        "request_id": rid,
        "messages": [{"role": "user", "content": [{"type": "text",
            "text": "Explain the entire theory of general relativity in extreme detail, step by step. Include all mathematical derivations."}]}],
        "params": {"thinking_mode": True, "max_thinking_tokens": 10, "max_response_tokens": 50, "max_tokens": 60},
    })

    comp = fix.wait_for_event("generation_complete")
    full_text = comp.get("full_text", "")

    ok("generation_complete received", True)
    ok("thinking_budget_burned_out is True", comp.get("thinking_budget_burned_out") is True)
    ok("full_text ends with </think>", full_text.strip().endswith("</think>"),
       f"full_text[-50:]={full_text[-50:]!r}")


def test_stop(fix: TestFixture):
    section("Test 5: Stop mid-generation")
    fix.clear_events()
    rid = fix.request_id()
    fix.emit("generate", {
        "request_id": rid,
        "messages": [{"role": "user", "content": [{"type": "text",
            "text": "Write a very long essay about the history of the Roman Empire."}]}],
        "params": {"thinking_mode": False, "max_tokens": 16384},
    })

    # Wait for at least a few tokens, then stop
    fix.wait_for_token_count(min_count=3, timeout=30)
    fix.emit("stop")

    stopped = fix.wait_for_event("generation_stopped")
    ok("generation_stopped received", True)
    ok("partial_text is non-empty", len(stopped.get("partial_text", "")) > 0)


def test_kv_cache_reset(fix: TestFixture):
    section("Test 6: KV cache reset")
    fix.clear_events()
    fix.emit("reset_kv_cache")
    result = fix.wait_for_event("kv_cache_reset", timeout=30)
    ok("kv_cache_reset received", True)


def test_efficiency_metrics(fix: TestFixture):
    section("Test 7: Efficiency metrics completeness")
    fix.clear_events()
    rid = fix.request_id()
    fix.emit("generate", {
        "request_id": rid,
        "messages": [{"role": "user", "content": [{"type": "text",
            "text": "Count from 1 to 5."}]}],
        "params": {"thinking_mode": False, "max_tokens": 256},
    })

    comp = fix.wait_for_event("generation_complete")

    required_fields = [
        "request_id", "full_text", "num_tokens", "tokens_per_second",
        "ttft", "full_generation_latency_sec", "generation_duration",
        "generated_tokens", "time_to_first_token",
        "llm_time_to_first_response_token",
        "thinking_time_to_first_token", "thinking_duration",
        "input_text_tokens", "input_audio_duration_sec",
    ]
    for field in required_fields:
        ok(f"field '{field}' present", field in comp, f"Missing: {field}")

    ok("num_tokens > 0", comp.get("num_tokens", 0) > 0)
    ok("ttft is not None", comp.get("ttft") is not None)
    ok("full_text has content", len(comp.get("full_text", "")) > 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    all_tests = {
        "thinking_on": test_thinking_on,
        "thinking_off": test_thinking_off,
        "thinking_prefix": test_thinking_prefix,
        "burnout": test_max_thinking_tokens_burnout,
        "stop": test_stop,
        "kv_cache_reset": test_kv_cache_reset,
        "metrics": test_efficiency_metrics,
    }

    tests_to_run = all_tests
    if args.test:
        invalid = set(args.test) - set(all_tests.keys())
        if invalid:
            print(f"Unknown test(s): {invalid}")
            print(f"Available: {list(all_tests.keys())}")
            sys.exit(1)
        tests_to_run = {k: v for k, v in all_tests.items() if k in args.test}

    print("=" * 60)
    print(" Qwen3-8B Socket.IO Server — Integration Tests")
    print("=" * 60)
    print(f" Server: {args.host}:{args.port}")
    print(f" Tests: {list(tests_to_run.keys())}")

    fix = TestFixture(args.host, args.port, timeout=args.timeout)
    print(f" Connected to server (sid={fix._sio.sid})\n")

    try:
        for name, test_fn in tests_to_run.items():
            try:
                test_fn(fix)
            except TimeoutError as e:
                print(f"  ❌ {name} — TIMEOUT: {e}")
                global _FAIL
                _FAIL += 1
            except Exception as e:
                print(f"  ❌ {name} — ERROR: {e}")
                import traceback
                traceback.print_exc()
                _FAIL += 1
    finally:
        fix.close()

    print(f"\n{'=' * 60}")
    print(f" Results: {_PASS} passed, {_FAIL} failed, {_PASS + _FAIL} total")
    print(f"{'=' * 60}")

    if _FAIL > 0:
        sys.exit(1)
    else:
        print(" All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
