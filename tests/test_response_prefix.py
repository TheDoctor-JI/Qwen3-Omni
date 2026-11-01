"""Generic prefills use real prompt/stream code without model weights."""
import ast
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment
from test_generation_error_reporting import load_server_functions, SERVER, Socket


@pytest.fixture
def server(monkeypatch):
    ns = load_server_functions()
    node = next(n for n in ast.parse(SERVER.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_prepare_inputs')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SERVER), 'exec'), ns)
    ns.update(_MODEL_IS_INSTRUCT=True, _MAX_NEW_MM_PER_REQUEST=1000,
              _build_messages=lambda payload: ([{'role': 'user', 'content': 'Hello'}], []),
              _process_mm_info_cached=lambda *args: (None, None, None, 0, []))
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=lambda **kw: kw))
    template = json.loads((Path(__file__).parent / 'fixtures' / 'instruct_chat_template.json').read_text())['chat_template']
    env = ImmutableSandboxedEnvironment()
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return env.from_string(template).render(messages=messages, **kwargs)
    return ns, Processor()


def payload(prefix='<plan>', **options):
    return {'request_id': 'prefill', 'params': dict(
        thinking_mode=False, response_prefix=prefix, **options)}


@pytest.mark.parametrize('prefix', ['<plan>', 'arbitrary prefill: ', ''])
def test_instruct_prompt_has_exact_prefix_without_native_empty_block(server, prefix):
    ns, processor = server
    inputs, _, _, _ = ns['_prepare_inputs'](processor, payload(prefix))
    assert inputs['prompt'].endswith('<|im_start|>assistant\n' + prefix)
    assert '<think>' not in inputs['prompt']
    assert 'enable_thinking' not in processor.kwargs


@pytest.mark.parametrize('options,error', [
    ({'response_prefix': None}, 'must be a string'),
    ({'response_prefix': 3}, 'must be a string'),
    ({'thinking_mode': True}, 'thinking and priming disabled'),
    ({'thinking_prefix': '<think>'}, 'thinking and priming disabled'),
    ({'prime_thinking': True}, 'thinking and priming disabled'),
])
def test_invalid_request_reports_normal_generation_error(server, options, error):
    ns, processor = server
    request = payload()
    request['params'].update(options)
    socket = Socket()
    asyncio.run(ns['_stream_generate'](socket, 'sid', None, processor, request))
    assert [name for name, _ in socket.events] == ['generation_error']
    assert error in socket.events[0][1]['error']


def test_thinking_checkpoint_cannot_use_prefix_even_when_thinking_disabled(server):
    ns, processor = server
    ns['_MODEL_IS_INSTRUCT'] = False
    socket = Socket()
    asyncio.run(ns['_stream_generate'](socket, 'sid', None, processor, payload()))
    assert socket.events[0][0] == 'generation_error'
    assert 'Instruct checkpoint' in socket.events[0][1]['error']


class Model:
    def __init__(self, chunks, end=None):
        self.chunks, self.end = chunks, end
        self.inputs = None

    async def generate(self, inputs, *args):
        self.inputs = inputs
        for i, text in enumerate(self.chunks):
            yield SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=tuple(range(i + 1)) if text else ())],
                                  finished=i == len(self.chunks) - 1 and self.end is None)
        if self.end:
            raise self.end


@pytest.mark.parametrize('prefix', ['<plan>', '[reason]'])
def test_stream_echoes_once_and_existing_parser_extracts_reason(server, prefix):
    from AudioLLMInterface.utils.in_response_cot import InResponseReasonParser
    closing = '</plan>' if prefix == '<plan>' else '[/reason]'
    continuation = ['Assess', 'Assess the context' + closing[:3], 'Assess the context' + closing + 'Hello']
    ns, processor = server
    socket, model = Socket(), Model(continuation)
    asyncio.run(ns['_stream_generate'](socket, 'sid', model, processor, payload(prefix)))
    assert model.inputs['prompt'].endswith(prefix)
    tokens = [data for name, data in socket.events if name == 'token']
    accumulated = ''
    parser = InResponseReasonParser(True, prefix, closing)
    action = ''
    for token in tokens:
        accumulated += token['delta']
        action += parser.feed(token['delta'])
        assert accumulated == token['full_text']
    assert accumulated == prefix + continuation[-1]
    assert accumulated.count(prefix) == 1
    assert action + parser.feed('', final=True) == 'Hello'
    assert parser.reason == 'Assess the context'
    final = socket.events[-1][1]
    assert final['full_text'] == accumulated
    assert final['generated_tokens'] == tokens[-1]['generated_tokens'] == 3


def test_empty_eos_still_echoes_prefix_without_fabricated_generated_tokens(server):
    ns, processor = server
    socket = Socket()
    asyncio.run(ns['_stream_generate'](socket, 'sid', Model(['']), processor, payload()))
    token = next(data for name, data in socket.events if name == 'token')
    assert token['delta'] == token['full_text'] == '<plan>'
    assert token['finished'] and token['generated_tokens'] == 0
    assert token['ttft'] is None
    assert socket.events[-1][1]['full_text'] == '<plan>'
    assert socket.events[-1][1]['generated_tokens'] == 0


@pytest.mark.parametrize('chunks,partial', [([], ''), (['Still planning'], '<plan>Still planning')])
def test_cancellation_reports_only_visible_prefixed_partial_text(server, chunks, partial):
    ns, processor = server
    socket = Socket()
    asyncio.run(ns['_stream_generate'](socket, 'sid', Model(chunks, asyncio.CancelledError()), processor, payload()))
    assert socket.events[-1][0] == 'generation_stopped'
    assert socket.events[-1][1]['partial_text'] == partial
    assert not any(name == 'generation_complete' for name, _ in socket.events)


def test_stream_failure_after_prefix_uses_existing_error_path(server):
    ns, processor = server
    socket = Socket()
    asyncio.run(ns['_stream_generate'](socket, 'sid', Model(['Assess'], RuntimeError('failed')), processor, payload()))
    assert socket.events[-1] == ('generation_error', {'request_id': 'prefill', 'error': 'failed'})
    assert [data['delta'] for name, data in socket.events if name == 'token'] == ['<plan>Assess']


def test_no_model_output_is_failure_instead_of_synthetic_success(server):
    ns, processor = server
    socket = Socket()
    asyncio.run(ns['_stream_generate'](socket, 'sid', Model([]), processor, payload()))
    assert socket.events[-1][0] == 'generation_error'
    assert not any(name in {'token', 'generation_complete'} for name, _ in socket.events)
