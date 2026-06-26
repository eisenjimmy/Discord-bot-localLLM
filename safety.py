"""Rate limiting and basic safety helpers."""

import os
import re
import time
from collections import defaultdict, deque
from typing import Optional

# Per-user rate limits: 1 request every 5 seconds, max 10 per minute
MIN_INTERVAL_SECONDS = 5
MAX_REQUESTS_PER_MINUTE = 10

_last_request: dict[str, float] = {}
_request_timestamps: dict[str, deque[float]] = defaultdict(deque)


def check_rate_limit(user_id: str) -> tuple[bool, Optional[str]]:
    """
    Return (allowed, error_message).
    Enforces 1 request per 5 seconds and 10 requests per minute per user.
    """
    now = time.monotonic()
    uid = str(user_id)

    last = _last_request.get(uid)
    if last is not None and now - last < MIN_INTERVAL_SECONDS:
        wait = MIN_INTERVAL_SECONDS - (now - last)
        return False, f"Please wait {wait:.0f} seconds before sending another request; I am rate-limiting requests to ensure server stability."

    timestamps = _request_timestamps[uid]
    # Drop timestamps older than 60 seconds
    while timestamps and now - timestamps[0] > 60:
        timestamps.popleft()

    if len(timestamps) >= MAX_REQUESTS_PER_MINUTE:
        return False, "You have exceeded the rate limit of 10 requests per minute. Please wait one minute before trying again."

    _last_request[uid] = now
    timestamps.append(now)
    return True, None


# Discord hard limit is 2000 chars — stay just under
MAX_REPLY_CHARS = int(os.getenv("MAX_REPLY_CHARS", "1900"))
MAX_REPLY_WORDS = int(os.getenv("MAX_REPLY_WORDS", "0"))  # 0 = no word cap


def truncate_reply(
    text: str,
    max_words: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> str:
    """Trim replies to fit Discord and optional word limits."""
    text = text.strip()
    if not text:
        return text

    char_limit = max_chars if max_chars is not None else MAX_REPLY_CHARS
    word_limit = max_words if max_words is not None else MAX_REPLY_WORDS

    if word_limit and word_limit > 0:
        words = text.split()
        if len(words) > word_limit:
            text = " ".join(words[:word_limit])
            if not text.endswith((".", "!", "?")):
                text += "..."

    if char_limit and len(text) > char_limit:
        text = text[: char_limit - 3].rstrip() + "..."

    return text


# Patterns that should be refused before hitting the LLM
_UNSAFE_PATTERNS = [
    r"\b(kill yourself|kys)\b",
    r"\b(doxx?|dox)\b",
    r"\b(child\s*porn|cp\b)",
    r"\b(nazi|genocide)\s+(how|guide|tutorial)",
]

_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS), re.IGNORECASE)


def is_unsafe_request(text: str) -> bool:
    """Quick pre-filter for obviously unsafe user input."""
    return bool(_UNSAFE_RE.search(text))


REFUSAL_MESSAGE = (
    "I must decline that request as it violates my safety parameters. Please ask something else, and I will be happy to assist you."
)

TIMEOUT_FALLBACK = (
    "I apologize, but my response generation has timed out. Could you please repeat your query?"
)

LLM_DOWN_FALLBACK = (
    "I am currently unable to establish a connection to the local language model server. Please try again shortly."
)

# Backwards-compatible alias
OLLAMA_DOWN_FALLBACK = LLM_DOWN_FALLBACK