"""Pure helpers for PredGen response-availability timing.

The authoritative candidate is a token-ID sequence.  These helpers decode
candidate prefixes only to identify the first token prefix that exposes actual
response content; character offsets are never used to splice token state.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence


PREDGEN_RESPONSE_TIMING_CONTRACT_VERSION = "predgen_response_availability_v1"


def _decode_candidate(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return str(
        tokenizer.decode(
            [int(token_id) for token_id in token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        or ""
    )


def response_content_is_available(
    candidate_text: str,
    *,
    starts_in_thinking: bool,
    open_tag: str = "<think>",
    close_tag: str = "</think>",
) -> bool:
    """Return whether decoded candidate text contains visible response content.

    ``starts_in_thinking`` covers Qwen templates that prefill ``<think>`` in
    the prompt, so the generated candidate may contain the closing tag without
    containing the opening tag itself.
    """
    text = str(candidate_text or "")
    if not text:
        return False

    if starts_in_thinking:
        close_index = text.find(close_tag)
        if close_index < 0:
            return False
        return bool(text[close_index + len(close_tag):].strip())

    open_index = text.find(open_tag)
    if open_index >= 0:
        close_index = text.find(close_tag, open_index + len(open_tag))
        if close_index < 0:
            return False
        return bool(text[close_index + len(close_tag):].strip())

    # Defensive support for a prompt-prefilled opening tag whose state was not
    # available to the caller.
    close_index = text.find(close_tag)
    if close_index >= 0:
        return bool(text[close_index + len(close_tag):].strip())

    stripped = text.lstrip()
    if not stripped:
        return False
    # Do not classify a control tag while it is still assembling.
    if open_tag.startswith(stripped):
        return False
    return True


def first_response_token_index(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    starts_in_thinking: bool,
    open_tag: str = "<think>",
    close_tag: str = "</think>",
) -> Optional[int]:
    """Return the zero-based first response token index, or ``None``.

    Availability is monotonic after a real response token appears, so a binary
    search finds the shortest token prefix that exposes response content.  This
    avoids repeatedly decoding every candidate prefix during long thinking
    traces.
    """
    ids = [int(token_id) for token_id in token_ids]
    if not ids:
        return None

    def _available(prefix_length: int) -> bool:
        return response_content_is_available(
            _decode_candidate(tokenizer, ids[:prefix_length]),
            starts_in_thinking=starts_in_thinking,
            open_tag=open_tag,
            close_tag=close_tag,
        )

    if not _available(len(ids)):
        return None

    low = 1
    high = len(ids)
    while low < high:
        middle = (low + high) // 2
        if _available(middle):
            high = middle
        else:
            low = middle + 1
    return low - 1
