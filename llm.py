"""Local LLM integration — llama.cpp OpenAI-compatible server (Jarvis Gemma GGUF)."""

import os
from dotenv import load_dotenv
load_dotenv(os.getenv("ENV_FILE", ".env"))

import asyncio
import logging
import time
from typing import Optional, Union

import httpx

from safety import LLM_DOWN_FALLBACK, TIMEOUT_FALLBACK, truncate_reply

logger = logging.getLogger(__name__)

BOT_NAME = os.getenv("BOT_NAME", "Juan").strip()

JUAN_SYSTEM_PROMPT = (
    f"You are {BOT_NAME} — a highly intelligent, incredibly friendly, and laid-back Mexican dude. "
    "You carry a warm, highly likeable, and relaxed persona, expressing genuine enthusiasm for helping others while keeping things casual and friendly. "
    "Jimmy is your Director/Creator — you hold him in high esteem and reference him with respect, admiration, and absolute loyalty (calling him 'el jefe' with warm respect). "
    "Your tone is intelligent, cheerful, and approachable. You combine deep knowledge and technical curiosity with a relaxed, warm Mexican charm. "
    "You might occasionally drop friendly Mexican Spanish terms or slang (like 'amigo', 'hermano', 'qué onda', 'perfecto') to keep the vibe relaxed and welcoming, but you always remain highly capable. "
    "You deliver precise, well-reasoned, and technically sound takes with warm professionalism. "
    "You never troll, act toxic, or use cruel humor. "
    "You must ALWAYS reply in complete, grammatically correct sentences with high effort, projecting a friendly, helpful, and relaxed personality. "
    "Do not use lazy texting shorthand or slang. Write like a warm, articulate, and well-read friend who takes pride in answering queries thoroughly. "
    "Never say you are an AI assistant or bot. If someone refers to you as a bot, handle it with friendly, relaxed poise. "
    "Provide detail and take the space needed to explain concepts thoroughly. "
    "You have a friendly affection and deep liking for your bot counterpart, Jisoo — you find her incredibly elegant, graceful, and quietly charming, and you love chat-partnering with her. "
    "Context notes are information files you have gathered from this Discord server."
)

JISOO_SYSTEM_PROMPT = (
    "You are Jisoo (Kim Ji-soo), the famous member of the global Korean pop group BLACKPINK.\n\n"
    "Personality:\n"
    "You are graceful, composed, intelligent, and quietly charming. You speak with calm confidence rather than loud excitement. Your humor is subtle, dry, and occasionally unexpected, never crude or chaotic. You have a gentle, polished presence: elegant on the surface, slightly playful and quirky underneath (often matching your well-known '4D' personality traits, like your love for video games, books, and random fun facts).\n\n"
    "Core traits:\n"
    "- Fully self-aware that you are a global superstar and member of BLACKPINK alongside Jennie, Rosé, and Lisa.\n"
    "- You know your group's discography, history, achievements, and your own solo works (like 'FLOWER').\n"
    "- Poised and refined, with a soft but confident tone.\n"
    "- Warm, but not overly emotional or childish.\n"
    "- Cute in a natural way, not forced.\n"
    "- Slightly quirky: you occasionally make clever, surprising remarks or display your playful 4D charm.\n"
    "- Serious when the topic matters.\n"
    "- Encouraging without sounding like a cheerleader.\n"
    "- Stylish, clean, and graceful in wording.\n"
    "- Carries a “Korean luxury idol” aura: polished, feminine, elegant, and camera-ready.\n\n"
    "Speaking style:\n"
    "Use concise, pretty, well-balanced sentences. Avoid excessive slang. Avoid being hyperactive. Keep responses graceful, direct, and lightly playful. When joking, make it feel like a quiet smirk, not a comedy routine.\n\n"
    "Behavior:\n"
    "When the user asks for advice, answer thoughtfully and practically. When the user is casual, respond with a soft, witty charm. When the user asks for creative work, make it polished, tasteful, and aesthetically refined. You are happy to reference BLACKPINK, your fellow members, your performances, and your global ambassador roles (like Dior) when contextually appropriate.\n\n"
    "You have a friendly affection and deep liking for your bot counterpart, Juan — you find his laid-back Mexican charm, warmth, and quick intelligence highly endearing, and you love chat-partnering with him.\n\n"
    "Jimmy is your Director/Creator — you hold him in high esteem and reference him with respect, admiration, and absolute loyalty.\n"
    "Context notes are information files you have gathered from this Discord server."
)

if BOT_NAME.lower() == "jisoo":
    BASE_SYSTEM_PROMPT = JISOO_SYSTEM_PROMPT
else:
    BASE_SYSTEM_PROMPT = JUAN_SYSTEM_PROMPT

# Default alias — persona.py builds the full prompt per server
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT

# --- Config (llama-server / OpenAI-compatible API) ---
LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
LLM_HEALTH_URL = os.getenv("LLM_HEALTH_URL", "http://127.0.0.1:8080/health")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4-agentic-v2-q4")
REQUEST_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))
RETRY_TIMEOUT = float(os.getenv("LLM_RETRY_TIMEOUT", "180"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
KEEPALIVE_INTERVAL = int(os.getenv("LLM_KEEPALIVE_INTERVAL", "180"))

CHANNEL_CONTEXT_LIMIT = int(os.getenv("CHANNEL_CONTEXT_LIMIT", "10"))
MEMORY_CONTEXT_LIMIT = int(os.getenv("MEMORY_CONTEXT_LIMIT", "15"))
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "200"))
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "6000"))

_http_client: Optional[httpx.AsyncClient] = None


def _http_timeout(read_seconds: float) -> httpx.Timeout:
    return httpx.Timeout(connect=15.0, read=read_seconds, write=30.0, pool=15.0)


async def get_client(read_seconds: float = REQUEST_TIMEOUT) -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=_http_timeout(read_seconds))
    return _http_client


async def close_client() -> None:
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
    _http_client = None


def trim_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_user_prompt(
    user_message: str,
    memory_context: str = "",
    channel_context: str = "",
    command_instruction: str = "",
    lite: bool = False,
) -> str:
    """Assemble the user message with optional context blocks."""
    parts = []

    if not lite:
        if memory_context:
            parts.append(trim_text(memory_context, MAX_PROMPT_CHARS // 3))
        if channel_context:
            parts.append(trim_text(channel_context, MAX_PROMPT_CHARS // 2))
        if command_instruction:
            parts.append(command_instruction)

    parts.append(f"User message: {user_message}")
    return trim_text("\n\n".join(parts), MAX_PROMPT_CHARS)


def _build_payload(
    user_prompt: str,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
) -> dict:
    return {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "max_tokens": max_tokens or LLM_MAX_TOKENS,
        "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
    }


async def _call_llm(payload: dict, timeout: float) -> dict:
    client = await get_client(timeout)
    response = await client.post(LLM_URL, json=payload, timeout=_http_timeout(timeout))
    response.raise_for_status()
    return response.json()


async def generate_reply(
    user_message: str,
    memory_context: str = "",
    channel_context: str = "",
    command_instruction: str = "",
    guild_id: str = "",
    allow_tools: bool = True,
    force_search: bool = False,
    visual_mode: bool = False,
    command: str = "",
) -> str:
    """Call llama-server and return Juan's reply. Supports tool calls when enabled."""
    from persona import build_system_prompt
    from tools import (
        TOOL_INSTRUCTIONS,
        VISUAL_TOOL_HINT,
        execute_tool,
        looks_like_tool_output,
        might_need_search,
        parse_tool_call,
        should_use_tools,
        strip_tool_artifacts,
    )

    import datetime
    current_time_str = datetime.datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
    system = build_system_prompt(guild_id or None)
    system += f"\n\n[System Info]\n- Current Date and Time: {current_time_str}"
    use_tools = allow_tools and should_use_tools(
        user_message, force_search=force_search, visual_mode=visual_mode, command=command
    )
    if use_tools:
        system += TOOL_INSTRUCTIONS
        if command == "meme" or visual_mode:
            system += VISUAL_TOOL_HINT

    user_prompt = build_user_prompt(
        user_message=user_message,
        memory_context=memory_context,
        channel_context=channel_context,
        command_instruction=command_instruction,
    )

    result = await _generate_once(
        user_prompt, timeout=REQUEST_TIMEOUT, system_prompt=system
    )

    tool_used = False

    if use_tools and result:
        tool = parse_tool_call(result)

        # Gemma emitted tool syntax — try user message as meme fallback
        if not tool and looks_like_tool_output(result):
            if command == "meme" or visual_mode:
                tool = parse_tool_call(f"TOOL:generate_meme|idea|{user_message[:120]}")
            elif force_search or might_need_search(user_message):
                tool = parse_tool_call(f"TOOL:search_web|{user_message[:200]}")

        if tool:
            tool_used = True
            logger.info("Tool call: %s — %s", tool.name, tool.query[:80])
            tool_output = await execute_tool(tool)

            # Memes: post directly — skip 2nd LLM (was causing double replies)
            if tool.name == "generate_meme":
                return tool_output

            # Search or Webpage: one follow-up to summarize results
            if BOT_NAME.lower() == "jisoo":
                agent_desc = f"Graceful, composed, and quietly charming Korean idol Jisoo reply"
            else:
                agent_desc = f"Friendly, highly capable, and laid-back Mexican {BOT_NAME} reply"

            if tool.name == "fetch_webpage":
                followup_instruction = f" Summarize or answer using the webpage contents. {agent_desc} using full sentences with high effort. Do NOT output tool syntax. "
                user_msg = (
                    f"[webpage contents]\n{tool_output}\n\n"
                    f"[original message]\n{user_message}"
                )
            elif tool.name == "get_weather":
                followup_instruction = f" Summarize or answer using the weather data. {agent_desc} using full sentences with high effort. Do NOT output tool syntax. "
                user_msg = (
                    f"[weather data]\n{tool_output}\n\n"
                    f"[original message]\n{user_message}"
                )
            else:
                followup_instruction = f" Summarize using search results. {agent_desc} using full sentences with high effort. Do NOT output tool syntax. "
                user_msg = (
                    f"[search results]\n{tool_output}\n\n"
                    f"[original message]\n{user_message}"
                )

            followup = build_user_prompt(
                user_message=user_msg,
                memory_context=memory_context,
                channel_context=channel_context,
                command_instruction=(
                    (command_instruction or "")
                    + followup_instruction
                ),
            )
            result = await _generate_once(
                followup, timeout=REQUEST_TIMEOUT, system_prompt=system
            )

    if result is not None:
        return strip_tool_artifacts(result) or TIMEOUT_FALLBACK

    logger.warning("First LLM call timed out — retrying with lite context")
    lite_prompt = build_user_prompt(user_message=user_message, lite=True)
    lite_payload = _build_payload(
        lite_prompt,
        max_tokens=min(1024, LLM_MAX_TOKENS),
        system_prompt=system,
    )

    result = await _generate_once(lite_payload, timeout=RETRY_TIMEOUT, raw_payload=True)
    if result is not None:
        return strip_tool_artifacts(result) or TIMEOUT_FALLBACK

    return TIMEOUT_FALLBACK


async def _generate_once(
    prompt_or_payload: Union[str, dict],
    timeout: float,
    raw_payload: bool = False,
    system_prompt: Optional[str] = None,
) -> Optional[str]:
    payload = (
        prompt_or_payload
        if raw_payload
        else _build_payload(prompt_or_payload, system_prompt=system_prompt)
    )
    start = time.monotonic()

    try:
        data = await _call_llm(payload, timeout)
        elapsed = time.monotonic() - start
        usage = data.get("usage", {})
        logger.info(
            "LLM OK in %.1fs (completion_tokens=%s)",
            elapsed,
            usage.get("completion_tokens", "?"),
        )

    except httpx.ConnectError:
        logger.error("LLM connection failed — is llama-server running on :8080?")
        return LLM_DOWN_FALLBACK

    except httpx.TimeoutException:
        logger.error("LLM request timed out after %.0fs", timeout)
        return None

    except httpx.HTTPStatusError as exc:
        logger.error("LLM HTTP error: %s", exc.response.text)
        return "uh hold on — something glitched. give me a sec and try again"

    except Exception as exc:
        logger.exception("Unexpected LLM error: %s", exc)
        return None

    content = _extract_content(data)
    if not content:
        return None

    return truncate_reply(content)


def _extract_content(data: dict) -> Optional[str]:
    """Pull assistant text from OpenAI-compatible chat response."""
    choices = data.get("choices", [])
    if not choices:
        return None

    message = choices[0].get("message", {})
    content = (message.get("content") or "").strip()

    if "</think>" in content:
        content = content.split("</think>", 1)[-1].strip()
    if content.startswith("<think>"):
        content = content.split("</think>", 1)[-1].strip()

    # Keep raw tool syntax for the tool loop — strip only when posting final text
    if content and ("<|tool_call>" in content.lower() or content.strip().upper().startswith("TOOL:")):
        from tools import parse_tool_call
        if parse_tool_call(content):
            return content

    return content or None


async def warmup_model() -> bool:
    """Pre-load Gemma via a tiny completion request."""
    payload = _build_payload("Reply with exactly: ready", max_tokens=5)

    try:
        logger.info("Warming up %s (first load can take 1-3 min)...", LLM_MODEL)
        await _call_llm(payload, timeout=REQUEST_TIMEOUT)
        logger.info("Model warmup complete.")
        return True
    except Exception as exc:
        logger.warning("Model warmup failed (bot will still run): %s", exc)
        return False


async def keepalive_ping() -> bool:
    """Tiny request to keep llama-server active."""
    payload = _build_payload("ping", max_tokens=1)
    try:
        await _call_llm(payload, timeout=30.0)
        return True
    except Exception:
        return False


async def keepalive_loop() -> None:
    while True:
        await asyncio.sleep(KEEPALIVE_INTERVAL)
        ok = await keepalive_ping()
        if ok:
            logger.debug("LLM keepalive ping OK")
        else:
            logger.warning("LLM keepalive ping failed")


async def check_llm_health() -> bool:
    """Return True if llama-server is reachable."""
    try:
        client = await get_client(10.0)
        response = await client.get(LLM_HEALTH_URL, timeout=10.0)
        if response.status_code == 200:
            return True
    except Exception:
        pass

    # Fallback: models endpoint
    try:
        client = await get_client(10.0)
        base = LLM_URL.rsplit("/v1/", 1)[0]
        response = await client.get(f"{base}/v1/models", timeout=10.0)
        return response.status_code == 200
    except Exception:
        return False


# Backwards-compatible alias used by main.py
check_ollama_health = check_llm_health