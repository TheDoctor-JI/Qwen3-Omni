from __future__ import annotations

import unittest
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from predgen_timing import (
    append_missing_thinking_close_token_ids,
    candidate_is_in_thinking,
    first_response_token_index,
    generation_starts_in_thinking,
    response_content_is_available,
    split_rendered_thinking_prefix_token_ids,
)


class _PieceTokenizer:
    def __init__(self, pieces: dict[int, str]):
        self._pieces = dict(pieces)

    def decode(self, token_ids, **_kwargs):
        return "".join(self._pieces[int(token_id)] for token_id in token_ids)

    def encode(self, text, **_kwargs):
        matches = [token_id for token_id, piece in self._pieces.items() if piece == text]
        if not matches:
            raise ValueError(f"piece not found: {text!r}")
        return [matches[0]]


class _CharacterTokenizer:
    def encode(self, text, **_kwargs):
        return [ord(char) for char in text]

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(int(token_id)) for token_id in token_ids)


class PredGenTimingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = _PieceTokenizer({
            1: "reasoning",
            2: "</think>",
            3: "  ",
            4: "answer",
            5: "<think>",
            6: "direct answer",
            7: "<thi",
            8: "</thi",
        })

    def test_prompt_prefilled_thinking_requires_content_after_close(self) -> None:
        self.assertIsNone(first_response_token_index(
            self.tokenizer, [1, 2, 3], starts_in_thinking=True,
        ))
        self.assertEqual(first_response_token_index(
            self.tokenizer, [1, 2, 3, 4], starts_in_thinking=True,
        ), 3)

    def test_delta_prefix_starts_inside_thinking_even_when_prompt_ends_in_text(self) -> None:
        self.assertTrue(generation_starts_in_thinking(
            "chat-template<think>prior reasoning",
            thinking_prefix="prior reasoning",
            thinking_mode=True,
            model_is_instruct=False,
        ))

    def test_normal_open_prompt_and_nonthinking_modes(self) -> None:
        self.assertTrue(generation_starts_in_thinking(
            "chat-template<think>\n",
            thinking_prefix="",
            thinking_mode=True,
            model_is_instruct=False,
        ))
        self.assertFalse(generation_starts_in_thinking(
            "chat-template<think>prior reasoning",
            thinking_prefix="prior reasoning",
            thinking_mode=False,
            model_is_instruct=False,
        ))
        self.assertFalse(generation_starts_in_thinking(
            "chat-template<think>prior reasoning",
            thinking_prefix="prior reasoning",
            thinking_mode=True,
            model_is_instruct=True,
        ))

    def test_generated_think_block_uses_first_post_think_content_token(self) -> None:
        self.assertEqual(first_response_token_index(
            self.tokenizer, [5, 1, 2, 4], starts_in_thinking=False,
        ), 3)

    def test_no_think_response_and_partial_open_tag(self) -> None:
        self.assertEqual(first_response_token_index(
            self.tokenizer, [6], starts_in_thinking=False,
        ), 0)
        self.assertIsNone(first_response_token_index(
            self.tokenizer, [7], starts_in_thinking=False,
        ))

    def test_whitespace_is_not_response_content(self) -> None:
        self.assertFalse(response_content_is_available(
            "   ", starts_in_thinking=False,
        ))

    def test_candidate_thinking_phase_follows_close_boundary(self) -> None:
        self.assertTrue(candidate_is_in_thinking(
            self.tokenizer, [1], starts_in_thinking=True,
        ))
        self.assertFalse(candidate_is_in_thinking(
            self.tokenizer, [1, 2], starts_in_thinking=True,
        ))
        self.assertTrue(candidate_is_in_thinking(
            self.tokenizer, [5, 1], starts_in_thinking=False,
        ))

    def test_append_close_is_idempotent(self) -> None:
        closed, appended, completed_partial = append_missing_thinking_close_token_ids(
            self.tokenizer, [1],
        )
        self.assertEqual(closed, [1, 2])
        self.assertEqual(appended, [2])
        self.assertFalse(completed_partial)
        closed_again, appended_again, _ = append_missing_thinking_close_token_ids(
            self.tokenizer, closed,
        )
        self.assertEqual(closed_again, closed)
        self.assertEqual(appended_again, [])

    def test_append_close_replaces_token_aligned_partial_marker(self) -> None:
        closed, appended, completed_partial = append_missing_thinking_close_token_ids(
            self.tokenizer, [1, 8],
        )
        self.assertEqual(closed, [1, 2])
        self.assertEqual(appended, [2])
        self.assertTrue(completed_partial)

    def test_hybrid_prefix_split_keeps_open_marker_in_base_prompt(self) -> None:
        tokenizer = _CharacterTokenizer()
        base_ids, candidate_ids = split_rendered_thinking_prefix_token_ids(
            tokenizer,
            "chat-template<model-open-think>prior reasoning",
            "prior reasoning",
        )
        self.assertEqual(
            tokenizer.decode(base_ids),
            "chat-template<model-open-think>",
        )
        self.assertEqual(tokenizer.decode(candidate_ids), "prior reasoning")

    def test_hybrid_prefix_split_rejects_unrendered_candidate(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not end"):
            split_rendered_thinking_prefix_token_ids(
                _CharacterTokenizer(),
                "chat-template-without-prefix",
                "prior reasoning",
            )


if __name__ == "__main__":
    unittest.main()
