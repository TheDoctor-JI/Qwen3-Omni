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
    ns.update(_MODEL_PROFILE=checkpoint, _MAX_NEW_MM_PER_REQUEST=1000,
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
    suffix = rendered[0].rsplit('<|im_start|>assistant\n', 1)[1]
    assert suffix.startswith('<think>\n\n</think>\n\n') == (checkpoint == 'thinking' and not thinking)


@pytest.mark.parametrize('path,expected', [
    ('./Qwen3-Omni-30B-A3B-Instruct', 'instruct'),
    ('./Qwen3-Omni-30B-A3B-Thinking', 'thinking'),
    ('/models/QWEN-THINKING-FP8', 'thinking'), ('org/model-INSTRUCT', 'instruct'),
])
def test_explicit_checkpoint_profiles(path, expected):
    assert load_server_functions()['_detect_checkpoint_profile'](path) == expected


@pytest.mark.parametrize('path', ['/models/qwen', '/models/thinking-instruct', ''])
def test_unsupported_profile_rejected_before_model_loading(path):
    ns = load_server_functions()
    node = next(n for n in ast.parse(SERVER.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_load_model_processor')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SERVER), 'exec'), ns)
    ns['_load_config'] = lambda _: {}
    with pytest.raises(ValueError, match='exactly one supported profile'):
        ns['_load_model_processor'](SimpleNamespace(checkpoint_path=path))


@pytest.fixture
def checkpoint_server(monkeypatch):
    ns = load_server_functions()
    node = next(n for n in ast.parse(SERVER.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_prepare_inputs')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SERVER), 'exec'), ns)
    ns.update(_MAX_NEW_MM_PER_REQUEST=1000,
              _build_messages=lambda payload: ([{'role': 'user', 'content': 'Hello'}], []),
              _process_mm_info_cached=lambda *args: (None, None, None, 0, []))
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(
        SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    rendered, options = [], []
    env = ImmutableSandboxedEnvironment()
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            template = json.loads((Path(__file__).parent / 'fixtures' /
                                   f"{ns['_MODEL_PROFILE']}_chat_template.json").read_text())['chat_template']
            prompt = env.from_string(template).render(messages=messages, **kwargs)
            rendered.append(prompt)
            options.append(kwargs)
            return prompt
    return ns, Processor(), rendered, options


def inline_request(opening='<think>', closing='</think>', seed=''):
    return {'request_id': 'inline', 'params': {
        'thinking_mode': False, 'response_prefix': opening + seed,
        'in_response_cot': {'enabled': True, 'open_tag': opening, 'close_tag': closing}}}


@pytest.mark.parametrize('profile', ['thinking', 'instruct'])
@pytest.mark.parametrize('opening,closing', [('<think>', '</think>'), ('<plan>', '</plan>'), ('[reason]', '[/reason]')])
@pytest.mark.parametrize('seed', ['', 'Let me see. '])
@pytest.mark.parametrize('limit', [None, 128])
def test_inline_prompt_has_exact_single_prefill(checkpoint_server, profile, opening, closing, seed, limit):
    ns, processor, _, options = checkpoint_server
    ns['_MODEL_PROFILE'] = profile
    request = inline_request(opening, closing, seed)
    if limit is not None:
        request['params']['in_response_cot']['max_tokens'] = limit
    inputs, _, _, _ = ns['_prepare_inputs'](processor, request)
    suffix = inputs['prompt'].rsplit('<|im_start|>assistant\n', 1)[1]
    assert suffix == opening + seed
    assert 'enable_thinking' not in options[-1]


def test_thinking_template_settings_do_not_leak_between_requests(checkpoint_server):
    ns, processor, _, options = checkpoint_server
    ns['_MODEL_PROFILE'] = 'thinking'
    ns['_prepare_inputs'](processor, inline_request())
    for contract in [None, {'enabled': False}]:
        params = {'thinking_mode': False}
        if contract is not None:
            params['in_response_cot'] = contract
        inputs, _, _, _ = ns['_prepare_inputs'](processor, {'params': params})
        assert inputs['prompt'].endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
        assert options[-1]['enable_thinking'] is False
    inputs, _, _, _ = ns['_prepare_inputs'](processor, {'params': {'thinking_mode': True}})
    assert options[-1]['enable_thinking'] is True
    assert inputs['prompt'].endswith('<|im_start|>assistant\n')
    ns['_prepare_inputs'](processor, inline_request())
    assert 'enable_thinking' not in options[-1]


@pytest.mark.parametrize('changes', [
    {'thinking_mode': True}, {'prime_thinking': True}, {'thinking_prefix': '<think>'},
    {'in_response_cot': None}, {'in_response_cot': {'enabled': False}},
    {'in_response_cot': {'enabled': 1}},
    {'in_response_cot': {'enabled': True, 'open_tag': '', 'close_tag': '</think>'}},
    {'response_prefix': 'no opening tag'}, {'response_prefix': '<think>closed</think>'},
])
def test_invalid_thinking_prefill_never_renders_or_generates(checkpoint_server, changes):
    ns, processor, rendered, _ = checkpoint_server
    ns['_MODEL_PROFILE'] = 'thinking'
    request = inline_request()
    request['params'].update(changes)
    socket = Socket()
    asyncio.run(ns['_stream_generate'](socket, 'sid', None, processor, request))
    assert [name for name, _ in socket.events] == ['generation_error']
    assert not rendered
    with pytest.raises(ValueError):
        ns['_prepare_inputs'](processor, request)
    assert not rendered


@pytest.mark.parametrize('mode', ['closed', 'plan_eos'])
def test_thinking_real_template_stream_and_repairs(checkpoint_server, mode):
    from test_inline_plan_continuation import Engine, output
    from AudioLLMInterface.utils.in_response_cot import InResponseReasonParser
    ns, processor, rendered, _ = checkpoint_server
    ns['_MODEL_PROFILE'] = 'thinking'
    call = '{"name":"lookup","arguments":{}}'
    action = call
    attempts = {
        'closed': [[output('Plan.</think>' + action)]],
        'plan_eos': [[output('Plan.')], [output(action)]],
    }
    engine, socket = Engine(attempts[mode]), Socket()
    asyncio.run(ns['_stream_generate'](socket, 'sid', engine, processor, inline_request(seed='Seed. ')))
    assert rendered[0].endswith('<|im_start|>assistant\n')
    assert engine.calls[0][0]['prompt'].endswith('<|im_start|>assistant\n<think>Seed. ')
    reason = InResponseReasonParser(True, '<think>', '</think>')
    accumulated, tool = '', ''
    for name, data in socket.events:
        if name == 'token':
            accumulated += data['delta']
            assert accumulated == data['full_text']
            tool += reason.feed(data['delta'])
    tool += reason.feed('', final=True)
    assert tool == call
    assert reason.reason == 'Seed. Plan.'
    assert accumulated == '<think>Seed. Plan.</think>' + action
    assert [name for name, _ in socket.events].count('generation_complete') == 1
    assert len(engine.calls) == (2 if mode == 'plan_eos' else 1)
