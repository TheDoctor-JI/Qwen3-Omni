from __future__ import annotations

import unittest
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from predgen_timing import first_response_token_index, response_content_is_available


class _PieceTokenizer:
    def __init__(self, pieces: dict[int, str]):
        self._pieces = dict(pieces)

    def decode(self, token_ids, **_kwargs):
        return "".join(self._pieces[int(token_id)] for token_id in token_ids)


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
        })

    def test_prompt_prefilled_thinking_requires_content_after_close(self) -> None:
        self.assertIsNone(first_response_token_index(
            self.tokenizer, [1, 2, 3], starts_in_thinking=True,
        ))
        self.assertEqual(first_response_token_index(
            self.tokenizer, [1, 2, 3, 4], starts_in_thinking=True,
        ), 3)

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


if __name__ == "__main__":
    unittest.main()
