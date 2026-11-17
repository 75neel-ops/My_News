"""Fetch and summarize news via Gemini with Google Search grounding.

Only links supplied by Google's grounding metadata are used. Results are
restricted to the source domains selected by the user.
"""
import re
import time
from datetime import date
from urllib.parse import urlparse

import requests
from google import genai
from google.genai import types

import config
import sources as src
from models import Item

# Accept the requested bold-headline format, while tolerating common variations
# returned by the model (numbered lists, bullets, and em/en dashes).
LINE_RE = re.compile(
    r"^\s*(?:[-*•]\s+|\d+[.)]\s+)?"
    r"\*\*(?P<title>.+?)\*\*\s*"
    r"(?:[—–:：-]\s*)?(?P<summary>.+?)\s*$"
)
PLAIN_LINE_RE = re.compile(
    r"^\s*(?:[-*•]\s+|\d+[.)]\s+)?"
    r"(?P<title>[^—–:：]{5,120})\s+[—–:：]\s+(?P<summary>.+?)\s*$"
)


class GeminiNewsClient:
    def __init__(self, api_key: str, model: str = config.MODEL):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self._redirect_cache = {}
        self.last_error = None

    def fetch_items(self, topic, group_sources, prefs) -> list:
        prompt = self._build_prompt(topic, group_sources, prefs)
        response = self._generate(prompt)
        return self._parse(response, group_sources, prefs.items_per_group)

    def _build_prompt(self, topic, group_sources, prefs) -> str:
        is_social = group_sources[0].group == src.SOCIAL
        kind = "posts, videos or discussions" if is_social else "news articles"
        site_list = "\n".join(
            f"- {s.name}: {', '.join(s.domains)}" for s in group_sources
        )
        return f"""You are a careful news curator. Today's date is {date.today():%d %B %Y}.

Find up to {prefs.items_per_group} distinct, noteworthy {kind} about "{topic}", {prefs.timeframe.phrase}.

Allowed sources ONLY:
{site_list}

Use Google Search to find real pages from the allowed sources.
Rules:
- Report only items actually found in search results. Never invent items.
- Do not repeat the same story twice.
- Output one item per line in this format:
  **Headline** — crisp 1-2 sentence summary (max 45 words)
- Write headlines and summaries in {prefs.language}.
- No introduction, no closing remarks, no URLs, no other text.
- If nothing relevant is found, output exactly: NO_RESULTS"""

    def _generate(self, prompt: str):
        cfg = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.2,
        )
        self.last_error = None
        for attempt in range(config.MAX_RETRIES):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=prompt, config=cfg
                )
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        return None

    def _parse(self, response, group_sources, limit) -> list:
        if response is None or not getattr(response, "candidates", None):
            return []

        candidate = response.candidates[0]
        text = (getattr(response, "text", None) or "").strip()
        metadata = getattr(candidate, "grounding_metadata", None)
        if not text or "NO_RESULTS" in text.upper() or metadata is None:
            return []

        chunks = getattr(metadata, "grounding_chunks", None) or []
        supports = getattr(metadata, "grounding_supports", None) or []
        parsed_lines = []
        offset = 0

        for line in text.splitlines():
            line_bytes = len(line.encode("utf-8"))
            match = LINE_RE.match(line) or PLAIN_LINE_RE.match(line)
            if match:
                parsed_lines.append(
                    (offset, offset + line_bytes,
                     match.group("title").strip().strip("*"),
                     match.group("summary").strip())
                )
            offset += line_bytes + 1

        items = []
        for start, end, title, summary in parsed_lines:
            url, source_name = self._link_for_span(
                start, end, supports, chunks, group_sources, title
            )
            if url:
                items.append(Item(title, summary, url, source_name))
            if len(items) >= limit:
                break
        return items

    def _link_for_span(self, start, end, supports, chunks, group_sources, headline=""):
        # First use the precise grounding references for this answer segment.
        indices = []
        for support in supports:
            segment = getattr(support, "segment", None)
            if segment is None:
                continue
            seg_start = getattr(segment, "start_index", None) or 0
            seg_end = getattr(segment, "end_index", None) or 0
            if seg_start < end and seg_end > start:
                indices.extend(getattr(support, "grounding_chunk_indices", None) or [])

        candidates = []
        for index in dict.fromkeys(indices):
            if 0 <= index < len(chunks):
                web_ref = getattr(chunks[index], "web", None)
                if web_ref and getattr(web_ref, "uri", None):
                    candidates.append(web_ref)

        # Some Gemini responses omit segment supports but still provide valid
        # grounding chunks. Use those only when their domains match our allowlist.
        if not candidates:
            for chunk in chunks:
                web_ref = getattr(chunk, "web", None)
                if web_ref and getattr(web_ref, "uri", None):
                    candidates.append(web_ref)

        headline_words = {
            word.lower() for word in re.findall(r"[\w]{4,}", headline)
        }
        ranked = []
        for web_ref in candidates:
            final_url, host = self._resolve(
                web_ref.uri, getattr(web_ref, "title", "") or ""
            )
            source = src.find_by_host(host, group_sources)
            if source is None:
                # If a redirect could not be resolved, try the original URL host.
                original_host = urlparse(web_ref.uri).netloc.lower().removeprefix("www.")
                source = src.find_by_host(original_host, group_sources)
                if source:
                    final_url = web_ref.uri
            if source is None:
                continue
            page_title = (getattr(web_ref, "title", "") or "").lower()
            overlap = sum(1 for word in headline_words if word in page_title)
            ranked.append((overlap, final_url, source.name))

        if not ranked:
            return None, None
        ranked.sort(key=lambda row: row[0], reverse=True)
        _, url, source_name = ranked[0]
        return url, source_name

    def _resolve(self, uri: str, title: str):
        """Resolve grounding redirects, retaining a safe fallback URL."""
        if uri in self._redirect_cache:
            return self._redirect_cache[uri]

        final = uri
        try:
            response = requests.get(
                uri, allow_redirects=True, timeout=8, stream=True,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            final = response.url
            response.close()
        except Exception:
            pass

        host = urlparse(final).netloc.lower().removeprefix("www.")
        if not host or "vertexaisearch" in host or host == "google.com":
            title_host = title.lower().removeprefix("www.")
            if "/" in title_host:
                title_host = title_host.split("/", 1)[0]
            if "." in title_host and " " not in title_host:
                host = title_host

        result = (final, host)
        self._redirect_cache[uri] = result
        return result
