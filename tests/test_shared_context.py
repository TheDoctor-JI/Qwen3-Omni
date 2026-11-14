"""Shared-view preparation and namespace lifecycle without loading a GPU model."""
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from test_generation_error_reporting import load_server_functions, load_app, SERVER
from mm_llm_interface.protocol import AudioReference


def functions():
    ns = load_server_functions()
    names = {'_build_messages', '_encode_shared_audio'}
    nodes = [n for n in ast.parse(SERVER.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SERVER), 'exec'), ns)
    return ns


def audio(tmp_path, start=0, end=160):
    path = tmp_path / 'pcm'
    path.write_bytes(np.arange(480, dtype='<i2').tobytes())
    return AudioReference('speech', dict(source_id='speech', start_frame=start, end_frame=end,
        sample_rate=16000, channels=1, encoding='s16le',
        blocks=[dict(path=str(path), length=960, capacity=960, block_id='block')])).to_wire()


def test_shared_view_maps_exact_samples_and_reuses_multiple_versions(tmp_path):
    ns, calls = functions(), []
    def encode(kind, samples):
        calls.append(samples.copy())
        return samples
    ns['_encode_single_mm_item'] = encode
    cache = ns['MmItemCache']()
    def prepare(item):
        messages, temporary = ns['_build_messages'](dict(protocol_version=1,
            context={'process_id': 'manager', 'interaction_id': 'dialogue'},
            messages=[dict(role='user', content=[item])]))
        assert not temporary
        return ns['_process_mm_info_cached'](messages, cache)
    first = audio(tmp_path, 80, 240)
    a = prepare(first)
    assert a[3] == 1
    assert np.array_equal(a[0][0], np.arange(80, 240, dtype=np.float32) / 32768)
    assert prepare(first)[3] == 0
    other = audio(tmp_path, 160, 320)
    assert other['view_key'] != first['view_key'] and prepare(other)[3] == 1
    assert prepare(first)[3] == 0 and len(cache) == 2 and len(calls) == 2
    cache.retire()
    with pytest.raises(ValueError, match='Retired'):
        prepare(first)


def test_concurrent_requests_prepare_one_view_once():
    ns = functions()
    cache = ns['MmItemCache']()
    count = []
    def encode():
        count.append(1)
        time.sleep(0.02)
        return object()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: cache.get_or_encode('speech', 'view', encode), range(4)))
    assert len(count) == 1 and sum(new for _, new in results) == 1
    assert all(value is results[0][0] for value, _ in results)


def test_retirement_fences_late_preparation():
    ns = functions()
    cache = ns['MmItemCache']()
    started, release = threading.Event(), threading.Event()
    def encode():
        started.set()
        assert release.wait(2)
        return object()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(cache.get_or_encode, 'speech', 'view', encode)
        assert started.wait(1)
        cache.retire()
        release.set()
        with pytest.raises(ValueError, match='Retired'):
            future.result()
    assert len(cache) == 0


def test_namespace_survives_generation_disconnect_and_owner_end_aborts_attached_work():
    async def run():
        calls, aborted = [], []
        class Model:
            async def abort(self, rid):
                aborted.append(rid)
        async def stream(sio, sid, model, processor, payload, session_cache):
            calls.append((payload['request_id'], session_cache))
            await asyncio.Event().wait()
        socket = load_app(Model(), stream)
        h = socket.handlers
        for sid in ('owner', 'a', 'b', 'other-owner', 'other'):
            await h['connect'](sid, {})
        context = dict(process_id='manager', interaction_id='one')
        other = dict(process_id='manager', interaction_id='two')
        assert (await h['register_context']('owner', context))['ok']
        assert not (await h['register_context']('a', context))['ok']
        assert (await h['register_context']('other-owner', other))['ok']
        async def generate(sid, rid, ctx):
            result = await h['generate'](sid, dict(request_id=rid, protocol_version=1, context=ctx))
            assert result['ok']
            await asyncio.sleep(0)
        await generate('a', 'first', context)
        await h['disconnect']('a')
        await generate('b', 'second', context)
        await generate('other', 'independent', other)
        assert calls[0][1] is calls[1][1] and calls[1][1] is not calls[2][1]
        assert not (await h['retire_context']('b', context))['ok']
        await h['disconnect']('owner')
        assert 'second' in aborted and 'independent' not in aborted
        result = await h['generate']('b', dict(request_id='late', protocol_version=1, context=context))
        assert not result['ok']
        assert not (await h['register_context']('b', context))['ok']
        await h['disconnect']('other-owner')
        await h['disconnect']('b')
        await h['disconnect']('other')
    asyncio.run(run())
