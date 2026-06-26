"""Web search — Jarvis-style Google Custom Search with DuckDuckGo fallback."""

import logging
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import httpx

logger = logging.getLogger(__name__)

GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", "")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "")
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
SEARCH_TIMEOUT = float(os.getenv("SEARCH_TIMEOUT", "20"))

# Pull Google keys from Jarvis .env if not set locally
_jarvis_env = Path(os.getenv("JARVIS_DIR", str(Path.home() / "Applications/Jarvis"))) / ".env"
if _jarvis_env.exists() and (not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_ENGINE_ID):
    for line in _jarvis_env.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key == "GOOGLE_SEARCH_API_KEY" and not GOOGLE_SEARCH_API_KEY:
            GOOGLE_SEARCH_API_KEY = val
        if key == "GOOGLE_SEARCH_ENGINE_ID" and not GOOGLE_SEARCH_ENGINE_ID:
            GOOGLE_SEARCH_ENGINE_ID = val


def _google_enabled() -> bool:
    return bool(GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID)


async def search_google(query: str, limit: int = SEARCH_MAX_RESULTS) -> list[dict]:
    """Google Custom Search JSON API (same endpoint as Jarvis ToolCatalog)."""
    url = (
        "https://www.googleapis.com/customsearch/v1"
        f"?key={quote_plus(GOOGLE_SEARCH_API_KEY)}"
        f"&cx={quote_plus(GOOGLE_SEARCH_ENGINE_ID)}"
        f"&q={quote_plus(query)}"
        f"&num={min(limit, 10)}"
    )
    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()

    items = data.get("items", [])
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": _strip_html(item.get("snippet", "")),
            "source": "Google",
        }
        for item in items[:limit]
    ]


def _get_ddgs():
    """Import DDGS from ddgs or legacy duckduckgo_search package."""
    try:
        from ddgs import DDGS
        return DDGS
    except ImportError:
        from duckduckgo_search import DDGS
        return DDGS


async def search_duckduckgo(query: str, limit: int = SEARCH_MAX_RESULTS) -> list[dict]:
    """DuckDuckGo via ddgs library (OpenJarvis fallback pattern)."""
    try:
        DDGS = _get_ddgs()
    except ImportError:
        logger.warning("ddgs not installed — pip install ddgs")
        return await _search_duckduckgo_html(query, limit)

    results: list[dict] = []
    try:
        raw = list(DDGS().text(query, max_results=limit))
        for r in raw:
            results.append(
                {
                    "title": r.get("title", "Untitled"),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                    "source": "DuckDuckGo",
                }
            )
    except Exception as exc:
        logger.warning("DDGS failed (%s), trying HTML fallback", exc)
        return await _search_duckduckgo_html(query, limit)

    return results


async def _search_duckduckgo_html(query: str, limit: int) -> list[dict]:
    """Jarvis-style DDG HTML scrape fallback."""
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    bot_name = os.getenv("BOT_NAME", "Juan").strip()
    headers = {"User-Agent": f"Mozilla/5.0 (compatible; {bot_name}Bot/1.0)"}

    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT, follow_redirects=True) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        html = response.text

    link_re = re.compile(
        r'class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    snippet_re = re.compile(
        r'class="result__snippet"[^>]*>(?P<snippet>.*?)</',
        re.IGNORECASE | re.DOTALL,
    )

    links = link_re.findall(html)
    snippets = snippet_re.findall(html)
    results: list[dict] = []

    for i, (raw_url, title) in enumerate(links[:limit]):
        snippet = snippets[i] if i < len(snippets) else ""
        results.append(
            {
                "title": _strip_html(title),
                "url": _decode_ddg_url(raw_url),
                "snippet": _strip_html(snippet),
                "source": "DuckDuckGo",
            }
        )

    return results


def _decode_ddg_url(url: str) -> str:
    """Extract real URL from DuckDuckGo redirect links."""
    if "uddg=" in url:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if "uddg" in params:
            return params["uddg"][0]
    return url


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def format_results(results: list[dict]) -> str:
    """Format search hits for LLM context."""
    if not results:
        return "No web results found."

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"{i}. {r['title']}\n"
            f"   URL: {r['url']}\n"
            f"   {r['snippet']}\n"
            f"   ({r['source']})"
        )
    return "\n\n".join(parts)


async def search_web(query: str, limit: Optional[int] = None) -> str:
    """
    Search the web — Google first (if configured), DuckDuckGo fallback.
    Same strategy as Jarvis ToolCatalog.SearchSingleWebAsync.
    """
    query = query.strip()
    if not query:
        return "Empty search query."

    limit = limit or SEARCH_MAX_RESULTS
    results: list[dict] = []

    if _google_enabled():
        try:
            results = await search_google(query, limit)
            logger.info("Google search returned %d results for: %s", len(results), query[:60])
        except Exception as exc:
            logger.warning("Google search failed (%s), falling back to DDG", exc)

    if not results:
        try:
            results = await search_duckduckgo(query, limit)
            logger.info("DDG search returned %d results for: %s", len(results), query[:60])
        except Exception as exc:
            logger.error("DuckDuckGo search failed: %s", exc)
            return f"Search failed: {exc}"

    return format_results(results)


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.result = []
        self.ignore = False
        self.ignore_tags = {'script', 'style', 'header', 'footer', 'nav', 'noscript', 'iframe'}

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.ignore_tags:
            self.ignore = True

    def handle_endtag(self, tag):
        if tag.lower() in self.ignore_tags:
            self.ignore = False

    def handle_data(self, data):
        if not self.ignore:
            text = data.strip()
            if text:
                self.result.append(text)


def extract_text_from_html(html: str) -> str:
    extractor = HTMLTextExtractor()
    extractor.feed(html)
    text = " ".join(extractor.result)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


async def fetch_webpage(url: str) -> str:
    """Fetch a URL and return a clean text representation of its contents."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = re.sub(r'^<|>$', '', url)
        if not url.startswith(("http://", "https://")):
            return "Invalid URL protocol. Only HTTP and HTTPS are supported."

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type and "text/plain" not in content_type:
                return f"Unsupported content type: {content_type}. Only HTML and plain text are supported."
                
            html = response.text
            text = extract_text_from_html(html)
            
            if len(text) > 8000:
                text = text[:8000] + "\n[Content truncated due to length limits]"
            return text if text else "Webpage contains no readable text."
            
    except Exception as exc:
        logger.error("Failed to fetch webpage %s: %s", url, exc)
        return f"Failed to retrieve webpage contents: {exc}"


async def get_weather(location: str = "") -> str:
    """Fetch current weather from wttr.in for a given location."""
    loc = quote_plus(location.strip()) if location.strip() else ""
    url = f"https://wttr.in/{loc}?format=4"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            text = response.text.strip()
            if "<html" in text.lower() or "not found" in text.lower():
                return f"Could not retrieve weather details for '{location}'."
            return text
    except Exception as exc:
        logger.error("Weather lookup failed: %s", exc)
        return f"Weather API error: unable to retrieve weather details."