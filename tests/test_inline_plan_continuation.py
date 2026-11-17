"""One logical planning stream can repair EOS once; disabled streams stay native."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test_generation_error_reporting import load_server_functions, load_app, Socket
from AudioLLMInterface.utils.in_response_cot import InResponseReasonParser


def output(text, ids=(1, 2), finish='stop', stop=None):
    return SimpleNamespace(finished=finish is not None, outputs=[SimpleNamespace(
        text=text, token_ids=ids, finish_reason=finish, stop_reason=stop)])


class Engine:
    def __init__(self, attempts):
        self.attempts, self.calls, self.aborted = attempts, [], []

    async def generate(self, inputs, params, request_id):
        attempt = len(self.calls)
        self.calls.append((inputs, params, request_id))
        for item in self.attempts[attempt]:
            if isinstance(item, Exception):
                raise item
            yield item

    async def abort(self, request_id):
        self.aborted.append(request_id)


def setup(opening='<think>', closing='</think>', **overrides):
    ns = load_server_functions()
    ns['_MODEL_PROFILE'] = 'instruct'
    prefix = opening + 'Let me see. '
    payload = {'request_id': 'request', 'params': dict(
        thinking_mode=False, response_prefix=prefix,
        in_response_cot={'enabled': True, 'open_tag': opening, 'close_tag': closing})}
    payload['params'].update(overrides)
    inputs = {'prompt': 'Original context\nassistant\n' + prefix,
              'multi_modal_data': {'audio': ['decoded-audio']},
              'mm_processor_kwargs': {'use_audio_in_video': True}}
    params = SimpleNamespace(max_tokens=20, temperature=0.6, top_p=0.95, top_k=20)
    ns['_prepare_inputs'] = lambda *args: (inputs, params, [], [{'item_id': 'audio'}])
    processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=151645))
    return ns, payload, processor, inputs, params


def run(engine, **kwargs):
    ns, payload, processor, inputs, params = setup(**kwargs)
    socket = Socket()
    asyncio.run(ns['_stream_generate'](socket, 'sid', engine, processor, payload))
    return socket, payload, inputs, params


@pytest.mark.parametrize('opening,closing', [('<think>', '</think>'), ('<plan>', '</plan>')])
@pytest.mark.parametrize('stop', [None, '<|im_end|>', 151645])
def test_eos_repair_streams_one_complete_parseable_action(opening, closing, stop):
    engine = Engine([[output('A finished plan.', (10, 151645), stop=stop)], [
        output('Hel', (20, 21), finish=None),
        output('Hello', (20, 21, 22, 151645)),
    ]])
    socket, payload, inputs, params = run(engine, opening=opening, closing=closing)
    prefix = payload['params']['response_prefix']
    expected = prefix + 'A finished plan.' + closing + 'Hello'
    tokens = [data for name, data in socket.events if name == 'token']
    accumulated = ''
    plan = InResponseReasonParser(True, opening, closing)
    speech = ''
    for token in tokens:
        accumulated += token['delta']
        assert accumulated == token['full_text']
        speech += plan.feed(token['delta'])
        assert token['request_id'] == 'request'
    speech += plan.feed('', final=True)
    assert speech == 'Hello' and not plan.pending
    assert plan.reason == 'Let me see. A finished plan.'
    assert accumulated == expected
    assert [token['finished'] for token in tokens] == [False, False, False, True]
    assert sum(name == 'generation_start' for name, _ in socket.events) == 1
    assert sum(name == 'items_cached' for name, _ in socket.events) == 1
    assert sum(name == 'generation_complete' for name, _ in socket.events) == 1
    assert socket.events[-1][1]['full_text'] == expected
    assert socket.events[-1][1]['generated_tokens'] == 6
    assert tokens[-1]['generated_tokens'] == 6
    assert engine.calls[0][0] is inputs
    continuation, sampling, rid = engine.calls[1]
    assert continuation['prompt'] == inputs['prompt'] + 'A finished plan.' + closing
    assert continuation['multi_modal_data'] is inputs['multi_modal_data']
    assert continuation['mm_processor_kwargs'] is inputs['mm_processor_kwargs']
    assert rid == 'request:plan-continuation'
    assert sampling.max_tokens == 18 and params.max_tokens == 20
    assert sampling.temperature == params.temperature
    assert not socket._qwen_engine_requests


@pytest.mark.parametrize('flag', [None, {'enabled': False}])
def test_disabled_requests_with_literal_thinking_tags_are_not_repaired(flag):
    engine = Engine([[output('Unclosed thought')]])
    socket, payload, _, _ = run(engine, in_response_cot=flag)
    assert len(engine.calls) == 1
    assert socket.events[-1][0] == 'generation_complete'
    assert socket.events[-1][1]['full_text'] == '<think>Let me see. Unclosed thought'
    assert not hasattr(socket, '_qwen_engine_requests')


def test_already_closed_plan_needs_no_continuation():
    engine = Engine([[output('Plan.</think>Hello')]])
    socket, _, _, _ = run(engine)
    assert len(engine.calls) == 1
    assert socket.events[-1][0] == 'generation_complete'


@pytest.mark.parametrize('first', [
    output('Incomplete', finish='length'),
    output('Incomplete', stop='some-other-stop'),
    output('Incomplete', ids=tuple(range(20))),
    output('Nested <think>thought'),
    output('Incomplete', finish='abort'),
])
def test_ineligible_unclosed_plan_fails_instead_of_completing(first):
    engine = Engine([[first]])
    socket, _, _, _ = run(engine)
    assert len(engine.calls) == 1
    assert socket.events[-1][0] == 'generation_error'
    assert not any(name == 'generation_complete' for name, _ in socket.events)
    assert not any(data.get('finished') for name, data in socket.events if name == 'token')


@pytest.mark.parametrize('contract', [True, {}, {'enabled': 1},
    {'enabled': True, 'open_tag': '', 'close_tag': '</think>'},
    {'enabled': True, 'open_tag': '<think>', 'close_tag': '<think>'},
    {'enabled': True, 'open_tag': '<plan>', 'close_tag': '</plan>'},
])
def test_invalid_contract_never_calls_engine(contract):
    engine = Engine([])
    socket, _, _, _ = run(engine, in_response_cot=contract)
    assert not engine.calls
    assert [name for name, _ in socket.events] == ['generation_error']


@pytest.mark.parametrize('attempts', [[[]], [[output('Plan')], []],
    [[output('Plan')], [RuntimeError('continuation failed')]],
    [[output('Partial', finish=None)]],
])
def test_empty_or_failed_attempt_has_one_error_and_no_completion(attempts):
    engine = Engine(attempts)
    socket, _, _, _ = run(engine)
    assert len(engine.calls) <= 2
    assert socket.events[-1][0] == 'generation_error'
    assert sum(name == 'generation_error' for name, _ in socket.events) == 1
    assert not any(name == 'generation_complete' for name, _ in socket.events)


@pytest.mark.parametrize('when', ['first', 'between', 'second'])
@pytest.mark.parametrize('event', ['stop', 'disconnect'])
def test_cancel_aborts_active_attempt_without_starting_another(when, event):
    async def scenario():
        ns, payload, processor, _, _ = setup()
        entered = asyncio.Event()

        class WaitingEngine(Engine):
            async def generate(self, inputs, params, rid):
                self.calls.append((inputs, params, rid))
                if when == 'first' or len(self.calls) == 2:
                    yield output('Still planning' if when == 'first' else 'Partial response', finish=None)
                    entered.set()
                    await asyncio.Event().wait()
                yield output('Plan')

        engine = WaitingEngine([])

        async def stream(socket, sid, model, unused, request, session_cache=None):
            if when == 'between':
                original_emit = socket.emit
                async def emit(name, data, **kwargs):
                    await original_emit(name, data, **kwargs)
                    if name == 'token' and data['full_text'].endswith('</think>'):
                        entered.set()
                        await asyncio.Event().wait()
                socket.emit = emit
            await ns['_stream_generate'](socket, sid, model, processor, request, session_cache)

        socket = load_app(engine, stream)
        await socket.handlers['connect']('sid', {})
        await socket.handlers['generate']('sid', payload)
        await asyncio.wait_for(entered.wait(), 2)
        if event == 'stop':
            ack = await socket.handlers['stop']('sid', {'request_id': 'request'})
            assert ack['ok']
        else:
            await socket.handlers['disconnect']('sid')
        assert len(engine.calls) == (2 if when == 'second' else 1)
        expected_id = 'request:plan-continuation' if when == 'second' else 'request'
        assert engine.aborted == [expected_id]
        assert not socket._qwen_engine_requests
        stopped = [data for name, data in socket.events if name == 'generation_stopped']
        assert len(stopped) == 1
        if when == 'between':
            assert stopped[0]['partial_text'].endswith('</think>')
        if when == 'first':
            assert stopped[0]['partial_text'].endswith('Still planning')
            assert '</think>' not in stopped[0]['partial_text']
        assert not any(name == 'generation_complete' for name, _ in socket.events)

    asyncio.run(scenario())


def test_temp_files_cleaned_once_after_both_attempts(tmp_path):
    ns, payload, processor, inputs, params = setup()
    path = tmp_path / 'media.wav'
    path.write_bytes(b'media')
    ns['_prepare_inputs'] = lambda *args: (inputs, params, [str(path)], [])
    engine = Engine([[output('Plan')], [output('Hello')]])
    asyncio.run(ns['_stream_generate'](Socket(), 'sid', engine, processor, payload))
    assert len(engine.calls) == 2
    assert not path.exists()


@pytest.mark.parametrize('opening,closing', [('<think>', '</think>'), ('<plan>', '</plan>')])
@pytest.mark.parametrize('repair_eos', [False, True])
@pytest.mark.parametrize('action', [
    '{"name":"set_language","arguments":{"lang":"mandarin"}}',
    '<tool_call>{"name":"set_language","arguments":{"lang":"mandarin"}}</tool_call>',
])
def test_direct_stream_completes_actual_mm_tool_pipeline(tmp_path, opening, closing, repair_eos, action):
    from AudioLLMInterface.tests.test_mm_in_response_cot import setup as client_setup
    from AudioLLMInterface.Shared_Components.AudioLLMGenerationTrajectory import GenerationTerminationState
    attempts = ([[output('A finished plan.')], [output(action)]] if repair_eos else
                [[output('A finished plan.' + closing + action)]])
    engine = Engine(attempts)
    socket, _, _, _ = run(engine, opening=opening, closing=closing)
    client, result = client_setup(tmp_path, tools=True)
    result.gen_config.plan_history_template = 'Latest plan: {plan}'
    client.mm_config['in_response_cot'] = {'open_tag': opening, 'close_tag': closing}
    client._maybe_wait_for_audio_commit = Mock()
    client._wait_for_reason_cancellation = Mock(side_effect=AssertionError('Must not wait'))
    def initiate(_):
        for name, data in socket.events:
            if name == 'token':
                result.stream_token_queue.put(data['delta'])
                if data['finished']:
                    result.stream_token_queue.put(None)
    client._initiate_qwen_stream = initiate
    client._text_and_audio_generation_thread(result)
    assert result.termination_state == GenerationTerminationState.COMPLETED
    assert len(result.tool_calls) == 1 and result.tool_calls[0].tool_name == 'set_language'
    assert result.in_response_reason == 'Let me see. A finished plan.'
    plans = [item for item in client.context_items if item.is_plan]
    assert len(plans) == 1 and plans[0].text == 'Latest plan: Let me see. A finished plan.'
    assert len(engine.calls) == (2 if repair_eos else 1)
    assert not engine.aborted


@pytest.mark.parametrize('split', range(1, len('Plan.</think><action>Hello</action>tail')))
def test_old_action_tags_are_passed_through_without_truncation_or_abort(split):
    raw = 'Plan.</think><action>Hello</action>tail'
    engine = Engine([[output(raw[:split], (1,), finish=None), output(raw, (1, 2))]])
    socket, payload, _, _ = run(engine)
    accumulated = ''
    for name, data in socket.events:
        if name == 'token':
            accumulated += data['delta']
            assert accumulated == data['full_text']
    assert accumulated == payload['params']['response_prefix'] + raw
    assert socket.events[-1][0] == 'generation_complete'
    assert socket.events[-1][1]['full_text'] == accumulated
    assert not engine.aborted and len(engine.calls) == 1


def test_legacy_action_text_inside_plan_does_not_infer_plan_closure():
    raw = 'Plan.<action>not a boundary</action>'
    engine = Engine([[output(raw)], [output('Hello')]])
    socket, payload, inputs, _ = run(engine)
    assert len(engine.calls) == 2
    assert engine.calls[1][0]['prompt'] == inputs['prompt'] + raw + '</think>'
    assert socket.events[-1][1]['full_text'] == payload['params']['response_prefix'] + raw + '</think>Hello'
    assert not engine.aborted


def test_error_after_plan_closure_is_not_repaired():
    engine = Engine([[output('Plan.</think>Partial', finish=None), RuntimeError('engine failed')]])
    socket, _, _, _ = run(engine)
    assert socket.events[-1][0] == 'generation_error'
    assert len(engine.calls) == 1
    assert not any(name == 'generation_complete' for name, _ in socket.events)
