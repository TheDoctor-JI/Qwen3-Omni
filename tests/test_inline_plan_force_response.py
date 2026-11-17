"""Forced response guidance stays inside repaired planning, never spoken output."""
import asyncio

import pytest

from test_inline_plan_token_budget import budget_setup, complete_text, piece
from test_inline_plan_continuation import Engine
from test_generation_error_reporting import Socket
from AudioLLMInterface.utils.in_response_cot import InResponseReasonParser


STATEMENT = 'Ok, I need to produce my response now.'


def run(attempts, statement=STATEMENT, **options):
    ns, payload, processor, inputs, params = budget_setup(**options)
    payload['params']['in_response_cot']['force_response_statement'] = statement
    engine, socket = Engine(attempts), Socket()
    asyncio.run(ns['_stream_generate'](socket, 'sid', engine, processor, payload))
    return engine, socket, payload, inputs


@pytest.mark.parametrize('repair', ['budget', 'eos'])
@pytest.mark.parametrize('opening,closing', [('<think>', '</think>'), ('[plan]', '[/plan]')])
def test_statement_precedes_injected_close_in_stream_and_continuation(repair, opening, closing):
    raw = 'Plan. fragment'
    engine, socket, payload, inputs = run(
        [[piece('Plan.'), piece(raw, None if repair == 'budget' else 'stop')],
         [piece('Hello', 'stop')]], limit=len(raw) if repair == 'budget' else 64,
        opening=opening, closing=closing)
    retained = 'Plan.' if repair == 'budget' else raw
    repaired = retained + '\n' + STATEMENT + '\n' + closing
    text, metrics = complete_text(socket)
    assert text == payload['params']['response_prefix'] + repaired + 'Hello'
    assert engine.calls[1][0]['prompt'] == inputs['prompt'] + repaired
    assert engine.aborted == (['request'] if repair == 'budget' else [])
    # Injected guidance consumes no model-generated tokens and stays private.
    assert metrics['generated_tokens'] == len(raw) + len('Hello')
    parser = InResponseReasonParser(True, opening, closing)
    speech = ''.join(parser.feed(data['delta']) for name, data in socket.events if name == 'token')
    assert speech + parser.feed('', final=True) == 'Hello'
    assert parser.reason.endswith(retained + '\n' + STATEMENT + '\n')


@pytest.mark.parametrize('limit', [None, 32])
def test_streamed_partial_close_is_held_until_guidance_can_precede_it(limit):
    engine, socket, payload, inputs = run(
        [[piece('Plan.</t'), piece('Plan.</thi', 'stop')], [piece('Hello', 'stop')]], limit=limit)
    repaired = 'Plan.\n' + STATEMENT + '\n</think>'
    assert complete_text(socket)[0] == payload['params']['response_prefix'] + repaired + 'Hello'
    assert engine.calls[1][0]['prompt'] == inputs['prompt'] + repaired


def test_natural_close_is_not_modified():
    raw = 'Plan.</think>Hello'
    engine, socket, payload, _ = run([[piece('Plan.</thi'), piece(raw, 'stop')]], limit=64)
    assert complete_text(socket)[0] == payload['params']['response_prefix'] + raw
    assert len(engine.calls) == 1 and not engine.aborted


def test_disabled_planning_does_not_insert_guidance():
    raw = 'Ordinary response'
    engine, socket, payload, _ = run([[piece(raw, 'stop')]], enabled=False)
    assert complete_text(socket)[0] == payload['params']['response_prefix'] + raw
    assert len(engine.calls) == 1 and not engine.aborted


def test_abort_does_not_insert_guidance_or_continue():
    engine, socket, _, _ = run([[piece('Plan', 'abort')]])
    assert len(engine.calls) == 1
    assert socket.events[-1][0] == 'generation_error'
    assert all(STATEMENT not in data.get('full_text', '') for _, data in socket.events)


@pytest.mark.parametrize('statement', [None, True, 42, ['respond']])
def test_invalid_statement_never_starts_engine(statement):
    engine, socket, _, _ = run([], statement=statement)
    assert not engine.calls
    assert [name for name, _ in socket.events] == ['generation_error']
