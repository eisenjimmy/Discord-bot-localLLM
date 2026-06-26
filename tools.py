"""Tool-call layer — Juan can search the web and drop memes/GIFs."""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from meme import create_meme, format_meme_reply, might_need_meme
from search import get_weather, search_web

logger = logging.getLogger(__name__)

TOOL_INSTRUCTIONS = (
    "\n\n--- Tools ---\n"
    "NEVER output <|tool_call> tags or XML. Use ONLY these plain-text formats:\n"
    "search_web — TOOL:search_web|your query\n"
    "fetch_webpage — TOOL:fetch_webpage|url\n"
    "get_weather — TOOL:get_weather|location\n"
    "generate_meme — TOOL:generate_meme|idea|describe the meme\n"
    "generate_meme gif — TOOL:generate_meme|gif|idea|describe it\n"
    "generate_meme manual — TOOL:generate_meme|drake|top text|bottom text\n"
    "Reply with ONLY the TOOL line — no other text — when using a tool.\n"
    "If the user asks you to search the web, search the internet, or look up information online, you MUST output a search_web tool call line.\n"
    "If the user asks you about the weather or temperature, you MUST output a get_weather tool call line with the location."
)


VISUAL_TOOL_HINT = (
    "\nFor memes/GIFs, use TOOL:generate_meme|idea|... — one line only. "
    "Do not use <|tool_call> or JSON wrappers."
)


@dataclass
class ToolCall:
    name: str
    query: str
    arg2: str = ""
    arg3: str = ""


# Gemma / llama.cpp native tool syntax
_GEMMA_TOOL_BLOCK = re.compile(
    r"<\|tool_call\>(.*?)(?:<\|/tool_call\>|$)",
    re.DOTALL | re.IGNORECASE,
)
_GEMMA_CALL_NAME = re.compile(
    r"call:\s*([a-zA-Z_]+)",
    re.IGNORECASE,
)
_JSON_OBJ = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _normalize_tool_name(raw: str) -> str:
    name = raw.lower().strip()
    name = name.replace("geenerate", "generate")
    if "meme" in name:
        return "generate_meme"
    if "search" in name:
        return "search_web"
    if "fetch" in name:
        return "fetch_webpage"
    if "weather" in name:
        return "get_weather"
    return name


def _parse_meme_tool_call(query: str, arg2: str, arg3: str) -> ToolCall:
    """Build generate_meme ToolCall from parsed parts."""
    if query.lower() == "idea":
        return ToolCall("generate_meme", "idea", arg2)
    if query.lower() == "gif" and arg2.lower() == "idea":
        return ToolCall("generate_meme", "gif_idea", arg3 or arg2)
    if query.lower() == "gif":
        return ToolCall("generate_meme", "gif", arg2, arg3)
    return ToolCall("generate_meme", query, arg2, arg3)


def _parse_pipe_tool(line: str) -> Optional[ToolCall]:
    """Parse TOOL:name|arg|... format."""
    if re.match(r"^TOOL:search_web\|", line, re.IGNORECASE):
        q = line.split("|", 1)[1].strip()
        return ToolCall("search_web", q) if q else None

    if re.match(r"^TOOL:fetch_webpage\|", line, re.IGNORECASE):
        url = line.split("|", 1)[1].strip()
        return ToolCall("fetch_webpage", url) if url else None

    if re.match(r"^TOOL:get_weather\|", line, re.IGNORECASE):
        loc = line.split("|", 1)[1].strip()
        return ToolCall("get_weather", loc)

    if not re.match(r"^TOOL:generate_meme\|", line, re.IGNORECASE):
        return None

    parts = line.split("|")
    if len(parts) >= 4 and parts[1].lower() == "gif" and parts[2].lower() == "idea":
        return ToolCall("generate_meme", "gif_idea", "|".join(parts[3:]).strip())
    if len(parts) >= 5 and parts[1].lower() == "gif":
        return ToolCall(
            "generate_meme", "gif_manual", parts[2].strip(),
            f"{parts[3].strip()}|{parts[4].strip()}",
        )
    if len(parts) >= 3 and parts[1].lower() == "gif":
        return ToolCall("generate_meme", "gif", parts[2].strip(), parts[3].strip() if len(parts) > 3 else "")
    if len(parts) >= 3 and parts[1].lower() == "idea":
        return ToolCall("generate_meme", "idea", "|".join(parts[2:]).strip())
    if len(parts) >= 4:
        return _parse_meme_tool_call(parts[1].strip(), parts[2].strip(), parts[3].strip())
    if len(parts) == 3:
        return _parse_meme_tool_call(parts[1].strip(), parts[2].strip(), "")

    return None


def _parse_gemma_tool_block(block: str) -> Optional[ToolCall]:
    """Parse <|tool_call>call:generate_meme{...} from Gemma."""
    name_match = _GEMMA_CALL_NAME.search(block)
    if not name_match:
        return None

    tool_name = _normalize_tool_name(name_match.group(1))
    json_match = _JSON_OBJ.search(block)

    if json_match and tool_name == "generate_meme":
        try:
            data = json.loads(json_match.group())
            if data.get("idea"):
                if str(data.get("format", "")).lower() == "gif":
                    return ToolCall("generate_meme", "gif_idea", str(data["idea"]))
                return ToolCall("generate_meme", "idea", str(data["idea"]))
            return ToolCall(
                "generate_meme",
                str(data.get("template", "drake")),
                str(data.get("top", "")),
                str(data.get("bottom", "")),
            )
        except json.JSONDecodeError:
            pass

    if json_match and tool_name == "search_web":
        try:
            data = json.loads(json_match.group())
            q = data.get("query") or data.get("q") or ""
            if q:
                return ToolCall("search_web", str(q))
        except json.JSONDecodeError:
            pass

    if json_match and tool_name == "fetch_webpage":
        try:
            data = json.loads(json_match.group())
            url = data.get("url") or data.get("link") or ""
            if url:
                return ToolCall("fetch_webpage", str(url))
        except json.JSONDecodeError:
            pass

    if json_match and tool_name == "get_weather":
        try:
            data = json.loads(json_match.group())
            loc = data.get("location") or data.get("query") or data.get("q") or ""
            return ToolCall("get_weather", str(loc))
        except json.JSONDecodeError:
            pass

    # Fallback: grab text after tool name as idea
    remainder = block[name_match.end():].strip(" {:\"'")
    if tool_name == "generate_meme" and remainder:
        return ToolCall("generate_meme", "idea", remainder[:200])

    if tool_name == "search_web" and remainder:
        return ToolCall("search_web", remainder[:200])

    return None


def looks_like_tool_output(text: str) -> bool:
    """True if model output is a tool attempt (parsed or not)."""
    if not text:
        return False
    t = text.strip()
    if t.upper().startswith("TOOL:"):
        return True
    if "<|tool_call>" in t.lower() or "call:generate" in t.lower() or "call:search" in t.lower() or "call:fetch" in t.lower():
        return True
    return False


def strip_tool_artifacts(text: str) -> str:
    """Remove raw tool-call syntax from user-facing text."""
    if not text:
        return text
    text = _GEMMA_TOOL_BLOCK.sub("", text)
    text = re.sub(r"call:\s*\w+\s*\{[^}]*\}", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^TOOL:\S+\|.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    return text.strip()


def parse_tool_call(text: str) -> Optional[ToolCall]:
    """Extract a tool call from model output (pipe format or Gemma tags)."""
    if not text:
        return None

    cleaned = text.strip()
    for line in cleaned.splitlines():
        line = line.strip()
        pipe = _parse_pipe_tool(line)
        if pipe:
            return pipe

    for block in _GEMMA_TOOL_BLOCK.findall(cleaned):
        gemma = _parse_gemma_tool_block(block)
        if gemma:
            return gemma

    # Whole text might be a gemma block without closing tag
    if "call:" in cleaned.lower():
        gemma = _parse_gemma_tool_block(cleaned)
        if gemma:
            return gemma

    try:
        if cleaned.startswith("{"):
            data = json.loads(cleaned)
            tool = _normalize_tool_name(str(data.get("tool", data.get("name", ""))))
            if tool == "search_web" and data.get("query"):
                return ToolCall("search_web", str(data["query"]).strip())
            if tool == "fetch_webpage" and data.get("url"):
                return ToolCall("fetch_webpage", str(data["url"]).strip())
            if tool == "get_weather":
                loc = data.get("location") or data.get("query") or data.get("q") or ""
                return ToolCall("get_weather", str(loc).strip())
            if tool == "generate_meme":
                if data.get("idea"):
                    if str(data.get("format", "")).lower() == "gif":
                        return ToolCall("generate_meme", "gif_idea", str(data["idea"]))
                    return ToolCall("generate_meme", "idea", str(data["idea"]))
                return ToolCall(
                    "generate_meme",
                    str(data.get("template", "drake")),
                    str(data.get("top", "")),
                    str(data.get("bottom", "")),
                )
    except json.JSONDecodeError:
        pass

    return None


async def execute_tool(call: ToolCall) -> str:
    """Run a parsed tool call and return text results."""
    if call.name == "search_web":
        return await search_web(call.query)

    if call.name == "fetch_webpage":
        from search import fetch_webpage
        return await fetch_webpage(call.query)

    if call.name == "get_weather":
        return await get_weather(call.query)

    if call.name == "generate_meme":
        if call.query == "idea":
            url, caption = await create_meme(idea=call.arg2)
        elif call.query == "gif_idea":
            url, caption = await create_meme(idea=call.arg2, prefer_gif=True)
        elif call.query == "gif_manual":
            top, _, bottom = call.arg3.partition("|")
            url, caption = await create_meme(
                template=call.arg2 or "oprah", top=top, bottom=bottom, prefer_gif=True,
            )
        elif call.query == "gif":
            url, caption = await create_meme(
                template=call.arg2 or "oprah", top=call.arg3, bottom="", prefer_gif=True,
            )
        else:
            url, caption = await create_meme(
                template=call.query, top=call.arg2, bottom=call.arg3,
            )
        if not url:
            return caption
        return format_meme_reply(url, caption)

    return f"Unknown tool: {call.name}"


_LIVE_HINTS = re.compile(
    r"\b(today|tonight|right now|currently|latest|recent|news|weather|forecast|temperature|temp|rain|snow|wind|score|"
    r"stock price|who is (the )?ceo|when did|what happened|how much is|"
    r"2025|2026|this week|yesterday|search|look up|google|websearch|web-search|"
    r"internet|online|browse|live|results|world cup|worldcup|match|game|standing|fixtures|playoff|bracket)\b",
    re.IGNORECASE,
)

def might_need_search(text: str) -> bool:
    return bool(_LIVE_HINTS.search(text))


_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def should_use_tools(
    user_message: str,
    force_search: bool = False,
    visual_mode: bool = False,
    command: str = "",
) -> bool:
    if force_search or might_need_search(user_message):
        return True
    if command == "meme" or visual_mode:
        return True
    if might_need_meme(user_message):
        return True
    if _URL_RE.search(user_message):
        return True
    return False