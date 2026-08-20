"""Pure helpers for PredGen response-availability timing.

The authoritative candidate is a token-ID sequence.  These helpers decode
candidate prefixes only to identify the first token prefix that exposes actual
response content; character offsets are never used to splice token state.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple


PREDGEN_RESPONSE_TIMING_CONTRACT_VERSION = "predgen_response_availability_v1"
PREDGEN_RESPONSE_TOKEN_DISTANCE_CONTRACT_VERSION = "response_tokens_v1"


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


def candidate_is_in_thinking(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    starts_in_thinking: bool,
    open_tag: str = "<think>",
    close_tag: str = "</think>",
) -> bool:
    """Return whether continuation from ``token_ids`` is still in thinking."""
    text = _decode_candidate(tokenizer, token_ids)
    if starts_in_thinking:
        return close_tag not in text
    open_index = text.find(open_tag)
    if open_index < 0:
        return False
    return text.find(close_tag, open_index + len(open_tag)) < 0


def append_missing_thinking_close_token_ids(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    close_tag: str = "</think>",
) -> Tuple[List[int], List[int], bool]:
    """Append exactly one close marker without retokenizing candidate content.

    A trailing partial marker is replaced only when it aligns to complete token
    boundaries.  If it cannot be repaired without changing earlier candidate
    tokens, fail explicitly instead of corrupting authoritative PredGen state.
    """
    ids = [int(token_id) for token_id in token_ids]
    text = _decode_candidate(tokenizer, ids)
    if close_tag in text:
        return ids, [], False

    partial_length = 0
    for length in range(min(len(close_tag) - 1, len(text)), 0, -1):
        if text.endswith(close_tag[:length]):
            partial_length = length
            break
    target_prefix = text[:-partial_length] if partial_length else text

    encode_kwargs = {"add_special_tokens": False}
    try:
        close_ids = [int(token_id) for token_id in tokenizer.encode(close_tag, **encode_kwargs)]
    except TypeError:
        close_ids = [int(token_id) for token_id in tokenizer.encode(close_tag)]
    if not close_ids:
        raise RuntimeError("PredGen tokenizer produced no token IDs for the thinking close tag")

    max_trim = min(16, len(ids)) if partial_length else 0
    for trim_count in range(max_trim + 1):
        kept = ids[:-trim_count] if trim_count else list(ids)
        if _decode_candidate(tokenizer, kept) != target_prefix:
            continue
        combined = kept + close_ids
        if _decode_candidate(tokenizer, combined) == target_prefix + close_tag:
            return combined, close_ids, bool(partial_length)
    raise RuntimeError(
        "PredGen could not append the thinking close tag without retokenizing candidate content"
    )
