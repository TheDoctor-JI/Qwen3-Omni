"""Planning-token interruption is a server continuation, never client cancellation."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test_inline_plan_continuation import setup, Engine, output
from test_generation_error_reporting import Socket, load_app


class Tokenizer:
    eos_token_id = 151645

    def decode(self, ids, **kwargs):
        return ''.join(chr(token) for token in ids if token != self.eos_token_id)


def piece(text, finish=None):
    return output(text, tuple(map(ord, text)), finish=finish)


def budget_setup(limit=4, total=256, enabled=True, opening='<think>', closing='</think>', profile='instruct'):
    contract = {'enabled': enabled, 'open_tag': opening, 'close_tag': closing, 'max_tokens': limit}
    ns, payload, processor, inputs, params = setup(
        opening=opening, closing=closing, in_response_cot=contract)
    ns['_MODEL_PROFILE'] = profile
    processor.tokenizer = Tokenizer()
    params.max_tokens = total
    return ns, payload, processor, inputs, params


def run_budget(attempts, **kwargs):
    ns, payload, processor, inputs, params = budget_setup(**kwargs)
    engine, socket = Engine(attempts), Socket()
    asyncio.run(ns['_stream_generate'](socket, 'sid', engine, processor, payload))
    return engine, socket, payload, inputs, params


def complete_text(socket):
    assert not any(name == 'generation_error' for name, _ in socket.events)
    accumulated = ''
    for name, data in socket.events:
        if name == 'token':
            accumulated += data['delta']
            assert accumulated == data['full_text']
    complete = [data for name, data in socket.events if name == 'generation_complete']
    assert len(complete) == 1 and complete[0]['full_text'] == accumulated
    assert sum(name == 'generation_start' for name, _ in socket.events) == 1
    return accumulated, complete[0]


@pytest.mark.parametrize('opening,closing', [('<think>', '</think>'), ('[plan]', '[/plan]')])
@pytest.mark.parametrize('profile', ['instruct', 'thinking'])
def test_exact_budget_aborts_engine_and_reserves_response_budget(opening, closing, profile):
    engine, socket, payload, inputs, params = run_budget(
        [[piece('P.'), piece('P. x'), RuntimeError('Do not consume after the cap')],
         [piece('Hello', 'stop')]], opening=opening, closing=closing, profile=profile)
    text, metrics = complete_text(socket)
    assert text == payload['params']['response_prefix'] + 'P.' + closing + 'Hello'
    assert engine.aborted == ['request']
    assert len(engine.calls) == 2
    continuation, remaining, rid = engine.calls[1]
    assert continuation['prompt'] == inputs['prompt'] + 'P.' + closing
    assert continuation['multi_modal_data'] is inputs['multi_modal_data']
    assert continuation['mm_processor_kwargs'] is inputs['mm_processor_kwargs']
    assert rid == 'request:plan-continuation'
    assert remaining.max_tokens == params.max_tokens - 4
    assert remaining.temperature == params.temperature
    assert metrics['generated_tokens'] == 4 + len('Hello')
    assert not socket._qwen_engine_requests


@pytest.mark.parametrize('finish', [None, 'stop'])
def test_batched_tokens_after_cap_are_not_published(finish):
    first = 'P. too long</think>discarded response'
    engine, socket, payload, _, _ = run_budget(
        [[piece('P.'), piece(first, finish)], [piece('Actual response', 'stop')]])
    text, metrics = complete_text(socket)
    assert text == payload['params']['response_prefix'] + 'P.</think>Actual response'
    assert 'too long' not in text and 'discarded response' not in text
    assert engine.aborted == (['request'] if finish is None else [])
    # Account for generated but discarded tokens in the shared total budget.
    assert engine.calls[1][1].max_tokens == 256 - len(first)
    assert metrics['generated_tokens'] == len(first) + len('Actual response')


@pytest.mark.parametrize('first,retained', [
    ('First sentence. unfinished', 'First sentence.'),
    ('An answer! unfinished', 'An answer!'),
    ('A question? unfinished', 'A question?'),
    ('完成。未完成', '完成。'),
    ('完成！未完成', '完成！'),
    ('完成？未完成', '完成？'),
    ('First. 再諗！Last? unfinished', 'First. 再諗！Last?'),
    ('No complete sentence', ''),
    ('Only a comma, or colon: fragment', ''),
    ('Complete.  \n', 'Complete.'),
])
def test_cap_sanitizes_stream_and_continuation_to_last_sentence(first, retained):
    # Character-by-character updates expose any premature streaming of the tail.
    chunks = [piece(first[:end]) for end in range(1, len(first) + 1)]
    engine, socket, payload, inputs, _ = run_budget(
        [chunks, [piece('Hello', 'stop')]], limit=len(first))
    expected_plan = payload['params']['response_prefix'] + retained
    assert complete_text(socket)[0] == expected_plan + '</think>Hello'
    assert engine.calls[1][0]['prompt'] == inputs['prompt'] + retained + '</think>'
    for name, data in socket.events:
        if name == 'token' and '</think>' not in data['full_text']:
            assert expected_plan.startswith(data['full_text'])
    assert engine.calls[1][1].max_tokens == 256 - len(first)


@pytest.mark.parametrize('finish', [None, 'length', 'stop'])
def test_cap_without_sentence_ending_preserves_prefill_and_drops_generated_fragment(finish):
    engine, socket, payload, inputs, _ = run_budget(
        [[piece('Pl'), piece('Plan', finish)], [piece('Hello', 'stop')]])
    assert complete_text(socket)[0] == payload['params']['response_prefix'] + '</think>Hello'
    assert engine.calls[1][0]['prompt'] == inputs['prompt'] + '</think>'


def test_natural_close_flushes_buffered_fragment_unchanged():
    raw = 'Complete. unfinished but valid</think>Hello'
    engine, socket, payload, _, _ = run_budget(
        [[piece('Complete. unfinished'), piece(raw, 'stop')]], limit=len(raw) + 1)
    assert complete_text(socket)[0] == payload['params']['response_prefix'] + raw
    first_token = next(data for name, data in socket.events if name == 'token')
    assert first_token['full_text'] == payload['params']['response_prefix'] + 'Complete.'
    assert len(engine.calls) == 1 and not engine.aborted


def test_eos_before_cap_flushes_buffered_fragment_without_sanitization():
    raw = 'Complete. unfinished'
    engine, socket, payload, inputs, _ = run_budget(
        [[piece(raw[:-1]), piece(raw, 'stop')], [piece('Hello', 'stop')]], limit=len(raw) + 1)
    assert complete_text(socket)[0] == payload['params']['response_prefix'] + raw + '</think>Hello'
    assert engine.calls[1][0]['prompt'] == inputs['prompt'] + raw + '</think>'


def test_buffering_keeps_engine_first_token_time_and_actual_token_metrics():
    ns, payload, processor, _, _ = budget_setup(limit=10)
    clock = [0.0]
    ns['time'] = SimpleNamespace(perf_counter=lambda: clock[0])
    class TimedEngine(Engine):
        async def generate(self, inputs, params, rid):
            self.calls.append((inputs, params, rid))
            if len(self.calls) == 1:
                clock[0] = 1.0
                yield piece('Un')
                clock[0] = 5.0
                yield piece('Unfinished')
            else:
                clock[0] = 8.0
                yield piece('Hello', 'stop')
    engine, socket = TimedEngine([]), Socket()
    asyncio.run(ns['_stream_generate'](socket, 'sid', engine, processor, payload))
    text, metrics = complete_text(socket)
    assert text == payload['params']['response_prefix'] + '</think>Hello'
    assert metrics['ttft'] == 1.0
    assert metrics['generated_tokens'] == 15
    assert metrics['total_time'] == 8.0
    assert metrics['tokens_per_second'] == round(15 / 8, 1)


def test_natural_close_disables_cap_for_later_response_tokens():
    first = 'P</think>'
    raw = first + 'A response much longer than the thinking token cap.'
    engine, socket, payload, _, _ = run_budget([[piece(first), piece(raw, 'stop')]], limit=len(first) + 1)
    assert complete_text(socket)[0] == payload['params']['response_prefix'] + raw
    assert len(engine.calls) == 1 and not engine.aborted


def test_natural_close_before_cap_in_a_large_batch_needs_no_recovery():
    raw = 'P</think>A response longer than the cap.'
    engine, socket, payload, _, _ = run_budget([[piece(raw, 'stop')]], limit=len('P</think>'))
    assert complete_text(socket)[0] == payload['params']['response_prefix'] + raw
    assert len(engine.calls) == 1 and not engine.aborted


@pytest.mark.parametrize('finish', [None, 'stop'])
def test_partial_closing_tag_completed_once(finish):
    first = 'Plan.</thi'
    engine, socket, payload, inputs, _ = run_budget(
        [[piece(first, finish)], [piece('Hello', 'stop')]], limit=len(first) if finish is None else 32)
    assert complete_text(socket)[0] == payload['params']['response_prefix'] + 'Plan.</think>Hello'
    assert engine.calls[1][0]['prompt'] == inputs['prompt'] + 'Plan.</think>'


def test_eos_before_cap_uses_same_single_continuation_allowance():
    engine, socket, payload, _, _ = run_budget([[piece('P', 'stop')], [piece('Response longer than cap', 'stop')]])
    assert complete_text(socket)[0] == payload['params']['response_prefix'] + 'P</think>Response longer than cap'
    assert len(engine.calls) == 2 and not engine.aborted


@pytest.mark.parametrize('limit', [0, -1, True, '128', 1.5, 256, 257])
def test_invalid_budget_never_starts_engine(limit):
    engine, socket, _, _, _ = run_budget([], limit=limit)
    assert not engine.calls
    assert [name for name, _ in socket.events] == ['generation_error']


def test_disabled_planning_ignores_budget():
    raw = 'Plan far beyond four tokens'
    engine, socket, payload, _, _ = run_budget([[piece(raw, 'stop')]], enabled=False)
    assert complete_text(socket)[0] == payload['params']['response_prefix'] + raw
    assert len(engine.calls) == 1 and not engine.aborted


def test_engine_error_does_not_trigger_budget_recovery():
    engine, socket, _, _, _ = run_budget([[piece('Pl'), RuntimeError('engine failed')]])
    assert socket.events[-1][0] == 'generation_error'
    assert len(engine.calls) == 1 and not engine.aborted


def test_external_abort_terminal_is_not_recovered_at_the_cap():
    engine, socket, _, _, _ = run_budget([[piece('Plan', 'abort')]])
    assert socket.events[-1][0] == 'generation_error'
    assert len(engine.calls) == 1


def test_consumed_total_budget_stops_active_engine_without_continuation():
    engine, socket, _, _, _ = run_budget([[piece('Plan' * 4)]], total=16)
    assert socket.events[-1][0] == 'generation_error'
    assert engine.aborted == ['request'] and len(engine.calls) == 1


def test_multibyte_token_boundary_backs_up_without_rewriting_sent_text():
    ns, _, _, _, _ = budget_setup()
    tokenizer = SimpleNamespace(decode=lambda ids, **kw: {0: '', 1: 'A', 2: 'A�', 3: 'A你'}[len(ids)])
    assert ns['_inline_plan_text_at_limit'](tokenizer, [1, 2, 3], 2, 'A你', 'A') == 'A'
    with pytest.raises(RuntimeError, match='revising streamed text'):
        ns['_inline_plan_text_at_limit'](tokenizer, [1, 2, 3], 2, 'A你', 'A你')


@pytest.mark.parametrize('when', ['before_close', 'after_close', 'response'])
@pytest.mark.parametrize('event', ['stop', 'disconnect'])
def test_client_cancellation_is_not_budget_recovery(when, event):
    async def scenario():
        ns, payload, processor, _, _ = budget_setup()
        entered = asyncio.Event()
        class WaitingEngine(Engine):
            async def generate(self, inputs, params, rid):
                self.calls.append((inputs, params, rid))
                if len(self.calls) == 1:
                    yield piece('P. x')
                    raise AssertionError('First attempt should end at cap')
                entered.set()
                await asyncio.Event().wait()
        engine = WaitingEngine([])
        async def stream(socket, sid, model, unused, request, session_cache=None):
            if when != 'response':
                emit_original = socket.emit
                async def emit(name, data, **kwargs):
                    await emit_original(name, data, **kwargs)
                    suffix = '</think>' if when == 'after_close' else 'P.'
                    if name == 'token' and data['full_text'].endswith(suffix):
                        entered.set()
                        await asyncio.Event().wait()
                socket.emit = emit
            await ns['_stream_generate'](socket, sid, model, processor, request, session_cache)
        socket = load_app(engine, stream)
        await socket.handlers['connect']('sid', {})
        await socket.handlers['generate']('sid', payload)
        await asyncio.wait_for(entered.wait(), 2)
        if event == 'stop':
            await socket.handlers['stop']('sid', {'request_id': 'request'})
        else:
            await socket.handlers['disconnect']('sid')
        assert len(engine.calls) == (2 if when == 'response' else 1)
        assert engine.aborted == ['request', 'request:plan-continuation' if when == 'response' else 'request']
        assert not socket._qwen_engine_requests
        stopped = [data for name, data in socket.events if name == 'generation_stopped']
        assert len(stopped) == 1
        if when == 'before_close':
            assert '</think>' not in stopped[0]['partial_text']
        else:
            assert '</think>' in stopped[0]['partial_text']
        assert not any(name == 'generation_complete' for name, _ in socket.events)
    asyncio.run(scenario())


def test_cancellation_during_internal_engine_abort_never_closes_or_restarts_plan():
    async def scenario():
        ns, payload, processor, _, _ = budget_setup()
        entered = asyncio.Event()
        class SlowAbortEngine(Engine):
            async def abort(self, rid):
                self.aborted.append(rid)
                if len(self.aborted) == 1:
                    entered.set()
                    await asyncio.Event().wait()
        engine = SlowAbortEngine([[piece('Plan')]])
        async def stream(socket, sid, model, unused, request, session_cache=None):
            await ns['_stream_generate'](socket, sid, model, processor, request, session_cache)
        socket = load_app(engine, stream)
        await socket.handlers['connect']('sid', {})
        await socket.handlers['generate']('sid', payload)
        await asyncio.wait_for(entered.wait(), 2)
        await socket.handlers['stop']('sid', {'request_id': 'request'})
        assert len(engine.calls) == 1 and engine.aborted == ['request', 'request']
        assert not any(name == 'generation_complete' for name, _ in socket.events)
        stopped = next(data for name, data in socket.events if name == 'generation_stopped')
        assert '</think>' not in stopped['partial_text']
        assert not socket._qwen_engine_requests
    asyncio.run(scenario())


@pytest.mark.parametrize('python_style', [False, True])
def test_budget_repaired_stream_executes_actual_client_tool_without_reason_leak(tmp_path, python_style):
    from AudioLLMInterface.tests.test_mm_in_response_cot import setup as client_setup
    from AudioLLMInterface.Shared_Components.AudioLLMGenerationTrajectory import GenerationTerminationState
    action = ('set_language(lang="mandarin")' if python_style else
              '{"name":"set_language","arguments":{"lang":"mandarin"}}')
    first = 'Plan. unfinished'
    _, socket, _, _, _ = run_budget([[piece('Plan. un'), piece(first)], [piece(action, 'stop')]],
                                   limit=len(first))
    client, result = client_setup(tmp_path, tools=True)
    result.gen_config.plan_history_template = 'Latest: {plan}'
    client.mm_config['in_response_cot'] = {'open_tag': '<think>', 'close_tag': '</think>'}
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
    assert result.in_response_reason == 'Let me see. Plan.'
    assert [item.text for item in client.context_items if item.is_plan] == ['Latest: Let me see. Plan.']
