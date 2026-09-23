"""Real checkpoint templates and server code; replace only GPU execution."""
import ast
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment
from test_generation_error_reporting import load_server_functions, SERVER, Socket, Model


@pytest.mark.parametrize('checkpoint', ['thinking', 'instruct'])
@pytest.mark.parametrize('thinking', [False, True])
@pytest.mark.parametrize('prime', [False, True])
@pytest.mark.parametrize('prefix', ['', '<think>\nConsider carefully.'])
def test_checkpoint_thinking_options(monkeypatch, checkpoint, thinking, prime, prefix):
    ns = load_server_functions()
    node = next(n for n in ast.parse(SERVER.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_prepare_inputs')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SERVER), 'exec'), ns)
    ns.update(_MODEL_IS_INSTRUCT=checkpoint == 'instruct', _MAX_NEW_MM_PER_REQUEST=1000,
              _build_messages=lambda payload: ([{'role': 'user', 'content': 'Hello'}], []),
              _process_mm_info_cached=lambda *args: (None, None, None, 0, []))
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=lambda **kw: kw))
    template = json.loads((Path(__file__).parent / 'fixtures' / f'{checkpoint}_chat_template.json').read_text())['chat_template']
    env = ImmutableSandboxedEnvironment()
    def raise_exception(message):
        raise ValueError(message)
    env.globals['raise_exception'] = raise_exception
    rendered = []
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            prompt = env.from_string(template).render(messages=messages, **kwargs)
            rendered.append(prompt)
            return prompt
    socket = Socket()
    asyncio.run(ns['_stream_generate'](socket, 'sid', Model(), Processor(), {
        'request_id': 'test', 'params': {'thinking_mode': thinking,
                                       'prime_thinking': prime, 'thinking_prefix': prefix}}))
    assert rendered
    assert [name for name, _ in socket.events] == ['generation_start', 'token', 'generation_complete']
    assert socket.events[-1][1]['full_text'].endswith('Hello')
