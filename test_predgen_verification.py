"""Exercise the actual verification control flow with a fake inference engine."""
import ast
import asyncio
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch


class VerificationTests(unittest.TestCase):
    def run_verification(self, k, candidate=(1, 2), finished=False, expired=False):
        # Extract the server's verification block to avoid importing/loading CUDA.
        tree = ast.parse(Path(__file__).with_name('socketio_server.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                  and n.name == '_predgen_style_generation')
        body = next(n.body for n in fn.body if isinstance(n, ast.Try))
        start = next(i for i, n in enumerate(body) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == '_deadline_reached()')
        end = 1 + next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'candidate_still_finished' for t in n.targets))
        wrapper = ast.parse('async def check():\n    pass').body[0]
        wrapper.body = body[start:end] + ast.parse('return locals()').body
        module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
        calls = []
        class Engine:
            async def generate(self, inputs, params, request_id):
                calls.append((inputs, params, request_id))
                yield types.SimpleNamespace(finished=True, prompt_token_ids=[9, *candidate],
                    prompt_logprobs=[{}, {1: 1}, {2: 7}])
        scope = dict(verification_top_k=k, candidate_token_ids=list(candidate),
            accepted_token_ids=list(candidate), _deadline_reached=lambda: expired,
            verification_completed=False, verification_duration=0.,
            verification_completed_elapsed=None, timed_out=False,
            rejection_index=None, candidate_ranks=[], model=Engine(),
            request_id='test', inputs={}, base_prompt_token_ids=[9],
            _predgen_inputs_with_token_ids=lambda inputs, ids: ids,
            _predgen_candidate_rank=lambda entry, token: entry.get(token),
            time=time, started_at=time.perf_counter(), tokenizer=None,
            first_response_token_index=lambda *a, **kw: 1 if len(a[1]) > 1 else None,
            starts_in_thinking=True, open_tag='<think>', close_tag='</think>',
            allow_summary_recovery=False, summary_threshold=.5, prior_finished=finished)
        wrapper.body = ast.parse('\n'.join(f'{key} = __initial[{key!r}]' for key in scope)).body + wrapper.body
        ast.fix_missing_locations(module)
        scope['__initial'] = dict(scope)
        exec(compile(module, '<server verification>', 'exec'), scope)
        with patch.dict(sys.modules, vllm=types.SimpleNamespace(SamplingParams=lambda **kw: kw)):
            result = asyncio.run(scope['check']())
        return result, calls

    def test_accept_all_finished_candidate_needs_no_inference(self):
        r, calls = self.run_verification(-1, finished=True)
        self.assertEqual(calls, [])
        self.assertTrue(r['candidate_still_finished'])
        self.assertEqual(r['accepted_ratio'], 1.)
        self.assertEqual(r['tokens_to_first_response'], 0)

    def test_accept_all_unfinished_candidate_can_continue(self):
        r, calls = self.run_verification(-1)
        self.assertEqual(calls, [])
        self.assertFalse(r['candidate_still_finished'])
        self.assertTrue(r['verification_completed'])

    def test_positive_k_still_rejects_suffix(self):
        r, calls = self.run_verification(3, finished=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(r['accepted_token_ids'], [1])
        self.assertEqual(r['rejection_index'], 1)
        self.assertFalse(r['candidate_still_finished'])

    def test_empty_candidate_still_requires_generation(self):
        r, calls = self.run_verification(-1, candidate=())
        self.assertEqual(calls, [])
        self.assertFalse(r['candidate_still_finished'])

    def test_expired_deadline_is_preserved(self):
        r, calls = self.run_verification(-1, expired=True)
        self.assertEqual(calls, [])
        self.assertTrue(r['timed_out'])
        self.assertFalse(r['verification_completed'])

if __name__ == '__main__':
    unittest.main()
