"""Real HTTP/Socket.IO lifecycle with GPU execution replaced by a tiny engine."""
import ast
import asyncio
import base64
from pathlib import Path
import threading
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest
import socketio

from test_generation_error_reporting import SERVER, load_server_functions
from qwen_service.config import startup_config, server_metadata


class Engine:
    def __init__(self):
        self.error = False
        self.abort_error = False
        self.requests = []
        self.aborted = []
        self.closed = False

    async def check_health(self):
        if self.error:
            raise RuntimeError('engine died')

    async def abort(self, request):
        self.aborted.append(request)
        if self.abort_error:
            raise RuntimeError('abort failed')

    async def generate(self, inputs, params, request):
        self.requests.append(request)
        yield SimpleNamespace(outputs=[SimpleNamespace(text='Hello', token_ids=[1])], finished=True)

    def shutdown(self):
        self.closed = True


def application(engine, prepare=None):
    ns = load_server_functions()
    ns.update(socketio=socketio, web=web)
    def default_prepare(processor, payload, cache):
        if payload.get('cache'):
            cache.get_or_encode('item', 'meta', lambda: b'cached')
        return {}, None, [], []
    ns['_prepare_inputs'] = prepare or default_prepare
    node = next(n for n in ast.parse(SERVER.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'create_socketio_app')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SERVER), 'exec'), ns)
    config, checkpoint, env = startup_config('thinking')
    metadata = server_metadata(config, checkpoint, 8899, 'persistent-test', env)
    return ns['create_socketio_app'](engine, None, metadata)


async def wait_status(client, predicate):
    for _ in range(100):
        response = await client.get('/status')
        data = await response.json()
        if predicate(data):
            return data
        await asyncio.sleep(.01)
    raise AssertionError(data)


def test_two_clients_keep_connections_until_disconnect_and_caches_are_isolated():
    async def run():
        engine = Engine()
        async with TestClient(TestServer(application(engine))) as http:
            url = str(http.make_url('/')).rstrip('/')
            a, b = socketio.AsyncClient(), socketio.AsyncClient()
            try:
                await a.connect(url, transports=['websocket'])
                await b.connect(url, transports=['websocket'])
                await a.call('generate', {'request_id': 'old', 'cache': True}, timeout=2)
                first = await wait_status(http, lambda s: s['session_cache_entries'] == 1 and s['active_generations'] == 0)
                # No engagement event is needed: connections persist until their owners disconnect.
                assert first['connected_clients'] == 2
                assert first['server_instance_id'] == 'persistent-test'
                await a.disconnect()
                await wait_status(http, lambda s: s['connected_clients'] == 1 and s['session_cache_entries'] == 0)
                await a.connect(url, transports=['websocket'])
                fresh = await wait_status(http, lambda s: s['connected_clients'] == 2)
                assert fresh['session_cache_entries'] == 0
                assert fresh['server_instance_id'] == first['server_instance_id']
                assert not engine.closed
                engine.error = True
                response = await http.get('/status')
                assert response.status == 503
                assert not (await response.json())['engine_healthy']
            finally:
                await a.disconnect()
                await b.disconnect()
        assert engine.closed
    asyncio.run(run())


def test_disconnect_during_thread_preprocessing_tracks_and_cleans_late_result(tmp_path):
    entered, release = threading.Event(), threading.Event()
    media = tmp_path / 'pending.wav'
    def prepare(*args):
        media.write_bytes(b'audio')
        entered.set()
        assert release.wait(5)
        return {}, None, [str(media)], []
    async def run():
        engine = Engine()
        async with TestClient(TestServer(application(engine, prepare))) as http:
            client = socketio.AsyncClient()
            await client.connect(str(http.make_url('/')).rstrip('/'), transports=['websocket'])
            try:
                await client.call('generate', {'request_id': 'cancelled'}, timeout=2)
                assert await asyncio.to_thread(entered.wait, 2)
                await wait_status(http, lambda s: s['pending_preprocessing'] == 1)
                await client.disconnect()
                await wait_status(http, lambda s: s['connected_clients'] == 0 and s['pending_preprocessing'] == 1)
                release.set()
                await wait_status(http, lambda s: s['pending_preprocessing'] == 0)
                assert not media.exists()
                assert not engine.requests
                assert engine.aborted == ['cancelled']
            finally:
                release.set()
                await client.disconnect()
    asyncio.run(run())


def test_disconnect_abort_failure_is_observable_and_prevents_reuse():
    async def run():
        engine = Engine()
        async with TestClient(TestServer(application(engine))) as http:
            client = socketio.AsyncClient()
            await client.connect(str(http.make_url('/')).rstrip('/'), transports=['websocket'])
            await client.call('generate', {'request_id': 'failed-abort'}, timeout=2)
            await wait_status(http, lambda s: len(engine.requests) == 1)
            engine.abort_error = True
            await client.disconnect()
            status = await wait_status(http, lambda s: s['engine_healthy'] is False)
            assert 'abort failed' in status['error']
            assert status['connected_clients'] == 0
            engine.abort_error = False
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['building', 'encoding'])
def test_preparation_exceptions_delete_created_files(tmp_path, failure, monkeypatch):
    ns = load_server_functions()
    import sys
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=lambda **kw: kw))
    for name in ('_build_messages', '_prepare_inputs'):
        node = next(n for n in ast.parse(SERVER.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SERVER), 'exec'), ns)
    media = tmp_path / 'created.wav'
    def save(*args):
        media.write_bytes(b'audio')
        return str(media)
    ns['_save_temp_b64'] = save
    ns['_MODEL_PROFILE'] = 'thinking'
    def encode(*args):
        raise ValueError('bad encoding')
    ns['_process_mm_info_cached'] = encode
    content = [{'type': 'audio', 'data': base64.b64encode(b'audio').decode()}]
    if failure == 'building':
        content.append(None)
    with pytest.raises((ValueError, AttributeError)):
        ns['_prepare_inputs'](None, {'messages': [{'role': 'user', 'content': content}]})
    assert not media.exists()
