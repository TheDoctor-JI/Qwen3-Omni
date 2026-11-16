"""Check actual checkpoint rendering without loading GPU dependencies."""
import ast
import json
from pathlib import Path

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment


SERVER = Path(__file__).resolve().parents[1] / 'socketio_server.py'
node = next(n for n in ast.parse(SERVER.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == '_tool_responses_as_system')
namespace = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(SERVER), 'exec'), namespace)
patch_template = namespace['_tool_responses_as_system']


@pytest.mark.parametrize('checkpoint', ['instruct', 'thinking'])
def test_tool_feedback_uses_system_turn_and_preserves_user_turns(checkpoint):
    template = json.loads((Path(__file__).parent / 'fixtures' /
                           f'{checkpoint}_chat_template.json').read_text())['chat_template']
    # Fixtures encode template newline escapes one level deeper than production.
    template = template.replace('\\\\n', '\\n')
    messages = [
        {'role': 'system', 'content': 'Instructions'},
        {'role': 'user', 'content': 'Question'},
        {'role': 'assistant', 'content': 'Looking up'},
        {'role': 'tool', 'content': 'First result'},
        {'role': 'tool', 'content': 'Second result'},
        {'role': 'user', 'content': 'Follow-up'},
        {'role': 'tool', 'content': 'Third result'},
    ]
    env = ImmutableSandboxedEnvironment()
    original = env.from_string(template).render(messages=messages, add_generation_prompt=True)
    rendered = env.from_string(patch_template(template)).render(
        messages=messages, add_generation_prompt=True)
    expected = original.replace('<|im_start|>user\n<tool_response>',
                                '<|im_start|>system\n<tool_response>')
    assert rendered == expected
    assert rendered.count('<|im_start|>system\n<tool_response>') == 2
    assert '<|im_start|>user\nQuestion<|im_end|>' in rendered
    assert '<|im_start|>user\nFollow-up<|im_end|>' in rendered
    assert rendered.count('<tool_response>') == 3
    assert rendered.endswith('<|im_start|>assistant\n')


def test_unrecognized_template_fails_explicitly():
    with pytest.raises(ValueError, match='Unsupported chat template'):
        patch_template('Unrecognized checkpoint template')
