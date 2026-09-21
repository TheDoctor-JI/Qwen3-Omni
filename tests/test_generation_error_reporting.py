"""Exercise server lifecycle code without importing or loading GPU models."""
import ast
import asyncio
import logging
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid


SERVER = Path(__file__).resolve().parents[1] / 'socketio_server.py'


def load_server_functions():
    tree = ast.parse(SERVER.read_text())
    names = {'MmItemCache', '_process_mm_info_cached', '_stream_generate'}
    nodes = [node for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name in names]
    namespace = dict(globals(), _logger=logging.getLogger('server-test'),
                     SERVER_CONFIG={}, _MM_MODALITIES={'audio', 'image', 'video'})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SERVER), 'exec'), namespace)
    return namespace


class Socket:
    def __init__(self):
        self.events = []

    async def emit(self, name, data, **kwargs):
        self.events.append((name, data))


class Model:
    def __init__(self):
        self.called = False

    async def generate(self, *args):
        self.called = True
        yield SimpleNamespace(outputs=[SimpleNamespace(text='Hello')], finished=True)


def test_cache_miss_reports_error_instead_of_silently_exiting():
    ns = load_server_functions()
    cache = ns['MmItemCache']()
    item = {'type': 'audio', 'item_id': 'missing', 'duration_ms': 100, 'sample_rate': 16000}
    messages = [{'role': 'user', 'content': [item]}]
    def prepare(*args):
        ns['_process_mm_info_cached'](messages, cache, 'request')
    ns['_prepare_inputs'] = prepare
    socket, model = Socket(), Model()
    asyncio.run(ns['_stream_generate'](socket, 'sid', model, None, {'request_id': 'request'}, cache))
    assert not model.called
    assert [name for name, _ in socket.events] == ['generation_error']
    assert socket.events[0][1]['request_id'] == 'request'
    assert 'Stub cache miss' in socket.events[0][1]['error']


def test_preparation_failure_reports_error():
    ns = load_server_functions()
    def prepare(*args):
        raise ValueError('invalid audio')
    ns['_prepare_inputs'] = prepare
    socket = Socket()
    asyncio.run(ns['_stream_generate'](socket, 'sid', Model(), None, {'request_id': 'request'}))
    assert socket.events == [('generation_error', {'request_id': 'request', 'error': 'invalid audio'})]


def test_success_preserves_cache_ack_order_and_cleanup(tmp_path):
    ns = load_server_functions()
    media = tmp_path / 'audio.wav'
    media.write_bytes(b'test')
    confirmed = [{'item_id': 'audio', 'meta_key': '100ms_16000hz'}]
    ns['_prepare_inputs'] = lambda *args: ({}, None, [str(media)], confirmed)
    socket = Socket()
    asyncio.run(ns['_stream_generate'](socket, 'sid', Model(), None, {'request_id': 'request'}))
    assert [name for name, _ in socket.events] == [
        'items_cached', 'generation_start', 'token', 'generation_complete']
    assert socket.events[-1][1]['full_text'] == 'Hello'
    assert not media.exists()


def test_cancel_during_preparation_reports_stopped():
    ns = load_server_functions()
    started, release = threading.Event(), threading.Event()
    def prepare(*args):
        started.set()
        assert release.wait(3)
        return {}, None, [], []
    ns['_prepare_inputs'] = prepare
    async def run():
        socket = Socket()
        task = asyncio.create_task(ns['_stream_generate'](
            socket, 'sid', Model(), None, {'request_id': 'request'}))
        try:
            assert await asyncio.to_thread(started.wait, 3)
            task.cancel()
            await task
            assert [name for name, _ in socket.events] == ['generation_stopped']
        finally:
            release.set()
    asyncio.run(run())


def load_app(model, stream):
    ns = load_server_functions()
    class FakeSocket(Socket):
        def __init__(self, **kwargs):
            super().__init__()
            self.handlers = {}
        def on(self, name):
            def register(fn):
                self.handlers[name] = fn
                return fn
            return register
        def attach(self, app):
            pass
    socket = FakeSocket()
    ns.update(socketio=SimpleNamespace(AsyncServer=lambda **kwargs: socket),
              web=SimpleNamespace(Application=lambda: SimpleNamespace(
                  router=SimpleNamespace(add_get=lambda *args: None))),
              _stream_generate=stream)
    node = next(n for n in ast.parse(SERVER.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'create_socketio_app')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SERVER), 'exec'), ns)
    ns['create_socketio_app'](model, None)
    return socket


def test_overlapping_generate_requests_remain_serialized():
    async def run():
        abort_entered, release_abort = asyncio.Event(), asyncio.Event()
        started, stopped, aborted = [], [], []
        class Engine:
            async def abort(self, request_id):
                aborted.append(request_id)
                if request_id == 'old':
                    abort_entered.set()
                    await release_abort.wait()
        async def stream(sio, sid, model, processor, payload, **kwargs):
            rid = payload['request_id']
            started.append(rid)
            try:
                await asyncio.Event().wait()
            finally:
                stopped.append(rid)
        socket = load_app(Engine(), stream)
        h = socket.handlers
        await h['connect']('sid', {})
        await h['generate']('sid', {'request_id': 'old'})
        await asyncio.sleep(0)
        a = asyncio.create_task(h['generate']('sid', {'request_id': 'A'}))
        await abort_entered.wait()
        b = asyncio.create_task(h['generate']('sid', {'request_id': 'B'}))
        await asyncio.sleep(0)
        assert started == ['old']
        assert not b.done()
        release_abort.set()
        await asyncio.gather(a, b)
        await asyncio.sleep(0)
        assert 'old' in stopped
        assert aborted == ['old', 'A']
        # A may be cancelled before its coroutine even starts.
        ack = await h['stop']('sid', {'request_id': 'B'})
        assert ack['ok'] and ack['request_id'] == 'B'
        assert 'B' in stopped
        await h['disconnect']('sid')
    asyncio.run(run())


def test_delayed_stop_does_not_cancel_replacement():
    async def run():
        aborted = []
        class Engine:
            async def abort(self, rid):
                aborted.append(rid)
        async def stream(*args, **kwargs):
            await asyncio.Event().wait()
        socket = load_app(Engine(), stream)
        h = socket.handlers
        await h['connect']('sid', {})
        await h['generate']('sid', {'request_id': 'new'})
        ack = await h['stop']('sid', {'request_id': 'old'})
        assert ack == {'ok': True, 'request_id': 'old', 'status': 'already_stopped'}
        assert aborted == []
        await h['disconnect']('sid')
        assert aborted == ['new']
    asyncio.run(run())


def test_abort_failure_rejects_ack_and_replacement():
    async def run():
        class Engine:
            async def abort(self, rid):
                raise RuntimeError('engine unavailable')
        async def stream(*args, **kwargs):
            await asyncio.Event().wait()
        socket = load_app(Engine(), stream)
        h = socket.handlers
        await h['connect']('sid', {})
        await h['generate']('sid', {'request_id': 'old'})
        ack = await h['stop']('sid', {'request_id': 'old'})
        assert ack['ok'] is False
        await h['generate']('sid', {'request_id': 'new'})
        assert socket.events[-1][0] == 'generation_error'
        assert socket.events[-1][1]['request_id'] == 'new'
        await h['disconnect']('sid')
    asyncio.run(run())


def test_socket_ack_waits_for_server_cancellation_cleanup():
    """Real Socket.IO transport, with only model execution replaced."""
    import socketio
    from aiohttp import web
    async def run():
        started, cleanup_started, finish_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()
        aborted = []
        class Engine:
            async def abort(self, rid):
                aborted.append(rid)
        async def stream(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await finish_cleanup.wait()
        ns = load_server_functions()
        ns.update(socketio=socketio, web=web, _stream_generate=stream)
        node = next(n for n in ast.parse(SERVER.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'create_socketio_app')
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SERVER), 'exec'), ns)
        runner = web.AppRunner(ns['create_socketio_app'](Engine(), None))
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        client = socketio.AsyncClient()
        try:
            await client.connect(f'http://127.0.0.1:{port}', transports=['websocket'])
            await client.call('generate', {'request_id': 'old'}, timeout=3)
            await asyncio.wait_for(started.wait(), 3)
            stop = asyncio.create_task(client.call('stop', {'request_id': 'old'}, timeout=3))
            await asyncio.wait_for(cleanup_started.wait(), 3)
            assert not stop.done()
            finish_cleanup.set()
            ack = await stop
            assert ack == {'ok': True, 'request_id': 'old', 'status': 'stopped'}
            assert aborted == ['old']
        finally:
            finish_cleanup.set()
            await client.disconnect()
            await runner.cleanup()
    asyncio.run(run())
