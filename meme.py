"""Meme and GIF generation via memegen.link API."""

import logging
import os
import random
import re
from typing import Optional
from urllib.parse import quote

import httpx

from llm import LLM_MAX_TOKENS, _build_payload, _call_llm, _extract_content

logger = logging.getLogger(__name__)

MEMEGEN_BASE = "https://api.memegen.link/images"
DEFAULT_TEMPLATE = "drake"
VISUAL_REPLY_CHANCE = float(os.getenv("VISUAL_REPLY_CHANCE", "0.15"))

# Templates that support animated GIF backgrounds
GIF_TEMPLATES = ["oprah", "iw", "drake", "buzz", "fine", "doge", "success"]

# Popular templates — validated against memegen.link
POPULAR_TEMPLATES = [
    "drake", "buzz", "blb", "fine", "mordor", "rollsafe", "pb",
    "chair", "disastergirl", "success", "both", "doge", "sparta", "wonka",
    "fry", "db", "ds", "vince", "money", "mb", "iw", "pigeon", "oprah",
    "custom", "buzz", "astronaut", "afraid", "bad", "biw", "cb", "gandalf",
]

MEME_IDEA_PROMPT = (
    "You create memes/GIFs using memegen.link. Given an idea, pick template + text. "
    "Reply ONLY:\n"
    "FORMAT: gif or png\n"
    "TEMPLATE: template_id\n"
    "TOP: top text\n"
    "BOTTOM: bottom text\n"
    "Use gif for reactions/energy (oprah, iw, doge). Use png for drake/buzz comparisons. "
    "Max 8 words per line. Keep it funny and relevant."
)


def encode_meme_text(text: str) -> str:
    """Encode text for memegen.link URL path (per their docs)."""
    if not text or not text.strip():
        return "_"

    text = text.strip()
    # Escape reserved characters first
    text = text.replace("_", "__")
    text = text.replace("-", "--")
    text = text.replace("?", "~q")
    text = text.replace("&", "~a")
    text = text.replace("%", "~p")
    text = text.replace("#", "~h")
    text = text.replace("/", "~s")
    text = text.replace("\\", "~b")
    text = text.replace("<", "~l")
    text = text.replace(">", "~g")
    text = text.replace('"', "''")
    text = text.replace("\n", "~n")
    text = text.replace(" ", "_")
    return text


def build_meme_url(
    template: str,
    top: str = "",
    bottom: str = "",
    fmt: str = "png",
    width: int = 600,
) -> str:
    """Build a memegen.link image URL."""
    template = (template or DEFAULT_TEMPLATE).strip().lower()
    top_enc = encode_meme_text(top)
    bottom_enc = encode_meme_text(bottom)
    url = f"{MEMEGEN_BASE}/{quote(template, safe='')}/{top_enc}/{bottom_enc}.{fmt}"
    if width:
        url += f"?width={width}"
    return url


def _parse_meme_idea_response(text: str) -> Optional[dict]:
    """Parse LLM meme idea response into template + texts."""
    if not text:
        return None

    fmt = "png"
    template = ""
    top = ""
    bottom = ""

    for line in text.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("FORMAT:"):
            fmt = line.split(":", 1)[1].strip().lower()
        elif upper.startswith("TEMPLATE:"):
            template = line.split(":", 1)[1].strip().lower()
        elif upper.startswith("TOP:"):
            top = line.split(":", 1)[1].strip()
        elif upper.startswith("BOTTOM:"):
            bottom = line.split(":", 1)[1].strip()

    if template and (top or bottom):
        return {"template": template, "top": top, "bottom": bottom, "fmt": fmt}
    return None


async def generate_meme_from_idea(idea: str, prefer_gif: bool = False) -> str:
    """Use LLM to pick template + text, return meme/GIF URL."""
    hint = " Prefer gif format for reactions." if prefer_gif else ""
    payload = _build_payload(
        f"Meme idea: {idea.strip()}.{hint}",
        max_tokens=min(200, LLM_MAX_TOKENS),
        system_prompt=MEME_IDEA_PROMPT,
        temperature=0.9,
    )

    try:
        data = await _call_llm(payload, timeout=120.0)
        parsed = _parse_meme_idea_response(_extract_content(data) or "")
        if parsed:
            fmt = parsed.get("fmt", "gif" if prefer_gif else "png")
            if fmt not in ("gif", "png", "webp"):
                fmt = "png"
            return build_meme_url(
                parsed["template"], parsed["top"], parsed["bottom"], fmt=fmt
            )
    except Exception as exc:
        logger.error("Meme idea LLM failed: %s", exc)

    fmt = "gif" if prefer_gif else "png"
    tmpl = "oprah" if prefer_gif else DEFAULT_TEMPLATE
    return build_meme_url(tmpl, "not this", idea[:40], fmt=fmt)


async def verify_meme_url(url: str) -> bool:
    """Check that memegen.link returns an image."""
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.head(url)
            if response.status_code == 200:
                return True
            # Some servers don't support HEAD — try GET
            response = await client.get(url)
            return response.status_code == 200
    except Exception:
        return False


async def create_meme(
    template: str = DEFAULT_TEMPLATE,
    top: str = "",
    bottom: str = "",
    idea: str = "",
    fmt: str = "png",
    prefer_gif: bool = False,
) -> tuple[str, str]:
    """
    Create a meme/GIF URL. Returns (url, caption).
    Uses idea + LLM if idea provided, otherwise template/top/bottom.
    """
    if idea.strip():
        url = await generate_meme_from_idea(idea, prefer_gif=prefer_gif or fmt == "gif")
        caption = f"made this for: {idea.strip()}"
    else:
        use_fmt = "gif" if (prefer_gif or fmt == "gif") else fmt
        url = build_meme_url(template, top, bottom, fmt=use_fmt)
        caption = f"**{template}** {'gif' if use_fmt == 'gif' else 'meme'}"

    if not await verify_meme_url(url):
        # Retry with drake if custom template failed
        if template != DEFAULT_TEMPLATE:
            url = build_meme_url(DEFAULT_TEMPLATE, top or "when template", bottom or "fails")
        if not await verify_meme_url(url):
            return "", "couldn't generate that meme — template might be wrong"

    return url, caption


def format_meme_reply(url: str, caption: str = "") -> str:
    """Format meme for Discord — URL auto-embeds as image."""
    if caption:
        return f"{caption}\n{url}"
    return url


_MEME_HINTS = re.compile(
    r"\b(meme|memegen|gif|giphy|make a meme|generate a meme|send a gif|reaction)\b",
    re.IGNORECASE,
)

_VISUAL_MOOD = re.compile(
    r"\b(lol|lmao|roast|mood|vibe|hot take|cringe|based|rip|bruh|dead|cap|"
    r"no shot|fr fr|lowkey|highkey|sus|ratio)\b",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_MEMEGEN_RE = re.compile(r"memegen\.link", re.IGNORECASE)


def might_need_meme(text: str) -> bool:
    return bool(_MEME_HINTS.search(text))


def might_want_visual(text: str) -> bool:
    """Heuristic — message could use a meme/GIF reply."""
    return bool(_MEME_HINTS.search(text) or _VISUAL_MOOD.search(text))


def has_visual_url(text: str) -> bool:
    return bool(_MEMEGEN_RE.search(text))


def should_add_visual(user_message: str, reply: str, visual_mode: bool) -> bool:
    """Decide if we should spontaneously attach a meme/GIF."""
    if has_visual_url(reply):
        return False
    if not visual_mode and not might_want_visual(user_message):
        return False
    return random.random() < VISUAL_REPLY_CHANCE


async def maybe_attach_visual(
    reply: str,
    user_message: str,
    visual_mode: bool = False,
) -> str:
    """Sometimes append a meme/GIF URL to a text reply."""
    if not should_add_visual(user_message, reply, visual_mode):
        return reply

    prefer_gif = bool(re.search(r"\bgif\b", user_message, re.IGNORECASE))
    idea = f"{user_message[:80]} — vibe: {reply[:100]}"
    url, _ = await create_meme(idea=idea, prefer_gif=prefer_gif)
    if not url:
        return reply

    logger.info("Attached spontaneous %s", "gif" if prefer_gif else "meme")
    return f"{reply}\n{url}"