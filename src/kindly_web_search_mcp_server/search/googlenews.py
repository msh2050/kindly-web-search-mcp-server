"""Google News RSS as a search source.

Why this exists: every *scraper* route to Google is blocked from some networks.
SearXNG ships its `google` engine as ``inactive: true``, forcing it on returns a
CAPTCHA, and Bing/Startpage/Brave are variously unreachable, CAPTCHA-locked or
rate-limited. Google News RSS is a published feed rather than a scrape, so it
answers reliably and cannot CAPTCHA out, and every item carries a real
``pubDate`` -- which is what makes recency usable for news queries.

The catch is that feed links are opaque ``news.google.com/rss/articles/...``
wrappers: they neither redirect nor embed the destination. :func:`resolve_url`
exchanges one through Google's ``batchexecute`` endpoint for the real article
URL. That endpoint is undocumented, so every failure path returns the wrapper
unchanged and the caller carries on with a slightly less useful link rather than
losing the result.

Resolution costs two extra requests per article, so callers should resolve only
the handful of results they intend to keep -- see
:func:`resolve_google_news_links`.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from ..models import WebSearchResult

LOGGER = logging.getLogger(__name__)

RSS_ENDPOINT = "https://news.google.com/rss/search"
BATCHEXECUTE_ENDPOINT = "https://news.google.com/_/DotsSplashUi/data/batchexecute"
WRAPPER_MARK = "news.google.com/rss/articles/"

# A browser UA: the feed itself is not fussy, but the wrapper page that carries
# the signature needed for resolution is served differently to obvious bots.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/139.0.0.0 Safari/537.36"
)

_ITEM_RE = re.compile(r"<item>(.*?)</item>", re.DOTALL)
_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.DOTALL)
_LINK_RE = re.compile(r"<link>(.*?)</link>", re.DOTALL)
_PUBDATE_RE = re.compile(r"<pubDate>(.*?)</pubDate>", re.DOTALL)
_SOURCE_RE = re.compile(r'<source url="(.*?)">(.*?)</source>', re.DOTALL)
_SIG_RE = re.compile(r'data-n-a-sg="([^"]+)"')
_TS_RE = re.compile(r'data-n-a-ts="([^"]+)"')
_TAG_RE = re.compile(r"<.*?>")


def _env(name: str, default: str) -> str:
    return (os.environ.get(name) or "").strip() or default


def _feed_params(query: str) -> dict[str, str]:
    """Build the feed query, honouring the configured locale.

    ``hl``/``gl``/``ceid`` decide which edition of Google News answers, which
    changes both language and which outlets appear.
    """
    lang = _env("KINDLY_GOOGLE_NEWS_LANG", "en-US")
    country = _env("KINDLY_GOOGLE_NEWS_COUNTRY", "US")
    return {
        "q": query,
        "hl": lang,
        "gl": country,
        "ceid": f"{country}:{lang.split('-')[0]}",
    }


def _parse_published(raw: str) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(raw.strip())
    except (TypeError, ValueError, IndexError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def parse_feed(xml: str, limit: int) -> list[tuple[WebSearchResult, datetime | None]]:
    """Turn feed XML into results paired with their publication time.

    The time is returned alongside rather than folded into
    :class:`WebSearchResult` so recency ordering stays available to the caller
    without widening the shared model that every provider returns.
    """
    out: list[tuple[WebSearchResult, datetime | None]] = []
    for chunk in _ITEM_RE.findall(xml)[:limit]:
        title_match = _TITLE_RE.search(chunk)
        link_match = _LINK_RE.search(chunk)
        if not (title_match and link_match):
            continue

        title = html.unescape(_TAG_RE.sub("", title_match.group(1))).strip()
        link = html.unescape(link_match.group(1)).strip()
        if not title or not link:
            continue

        source_match = _SOURCE_RE.search(chunk)
        publisher = html.unescape(source_match.group(2)).strip() if source_match else ""
        # Feed titles end in " - Publisher"; strip it and reuse the publisher as
        # the snippet, which is the only descriptive text the feed offers.
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -(len(publisher) + 3)].strip()

        date_match = _PUBDATE_RE.search(chunk)
        published = _parse_published(date_match.group(1)) if date_match else None
        snippet = publisher or "Google News"
        if published:
            snippet = f"{snippet} — {published.strftime('%Y-%m-%d %H:%M')}"

        out.append(
            (
                WebSearchResult(title=title, link=link, snippet=snippet, page_content=""),
                published,
            )
        )
    return out


async def resolve_url(url: str, client: httpx.AsyncClient) -> str:
    """Exchange a Google News wrapper URL for the real article URL.

    Returns the input unchanged on any failure: an unresolved wrapper still
    identifies the story, so a broken resolution should degrade the result
    rather than discard it.
    """
    if WRAPPER_MARK not in url:
        return url
    try:
        article_id = url.split("/articles/")[1].split("?")[0]
        # follow_redirects is explicit: the wrapper 302s to itself with locale
        # params appended, and a shared client that does not follow redirects
        # yields an empty body -- no signature, no resolution, silent fallback
        # to the unusable wrapper link.
        page = await client.get(url, timeout=20, follow_redirects=True,
                                headers={"User-Agent": USER_AGENT})
        sig = _SIG_RE.search(page.text)
        ts = _TS_RE.search(page.text)
        if not (sig and ts):
            LOGGER.debug("No signature on Google News wrapper; keeping %s", url)
            return url

        inner = json.dumps(
            [
                "garturlreq",
                [
                    ["X", "X", ["X", "X"], None, None, 1, 1, "US:en", None, 1,
                     None, None, None, None, None, 0, 1],
                    "X", "X", 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0,
                ],
                article_id,
                int(ts.group(1)),
                sig.group(1),
            ]
        )
        resp = await client.post(
            BATCHEXECUTE_ENDPOINT,
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            },
            data={"f.req": json.dumps([[["Fbv4je", inner, None, "generic"]]])},
            timeout=20,
        )
        for candidate in re.findall(r'https?://[^"\\\s]+', resp.text):
            if "google.com" not in candidate and "gstatic" not in candidate:
                return candidate
    except Exception as exc:  # noqa: BLE001 -- deliberate
        # batchexecute is an undocumented internal endpoint: it can fail in
        # ways there is no point enumerating, and none of them should cost the
        # caller a result. Falling back to the wrapper link is always valid.
        LOGGER.warning("Google News URL resolution failed (%s); keeping wrapper",
                       type(exc).__name__)
    return url


async def resolve_google_news_links(
    results: list[WebSearchResult], client: httpx.AsyncClient
) -> list[WebSearchResult]:
    """Resolve every wrapper link in ``results``, concurrently.

    Call this *after* the result set has been trimmed to what will be returned:
    each resolution costs two requests, so resolving a full feed would be far
    more traffic than the answer needs.
    """
    targets = [r for r in results if WRAPPER_MARK in r.link]
    if not targets:
        return results

    resolved = await asyncio.gather(
        *(resolve_url(r.link, client) for r in targets), return_exceptions=True
    )
    for result, link in zip(targets, resolved):
        if isinstance(link, str) and link:
            result.link = link
    return results


async def search_google_news(
    query: str,
    *,
    num_results: int,
    http_client: httpx.AsyncClient | None = None,
    resolve_links: bool = True,
) -> list[WebSearchResult]:
    """Fetch headlines for ``query`` from Google News RSS.

    Args:
        query: The search query.
        num_results: Maximum number of headlines to return.
        http_client: Client to reuse; a short-lived one is created when omitted.
        resolve_links: Whether to exchange wrapper URLs for real article URLs.
            Leave this off when the caller will merge these results with others
            and trim afterwards, then resolve only the survivors.

    Returns:
        Up to ``num_results`` headlines, newest first.
    """
    if not query.strip() or num_results < 1:
        return []

    async def _run(client: httpx.AsyncClient) -> list[WebSearchResult]:
        resp = await client.get(
            RSS_ENDPOINT,
            params=_feed_params(query),
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        resp.raise_for_status()
        # Over-fetch a little so recency sorting has something to choose from.
        pairs = parse_feed(resp.text, max(num_results * 4, 40))
        pairs.sort(key=lambda pair: pair[1] or datetime.min.replace(tzinfo=UTC),
                   reverse=True)
        results = [result for result, _ in pairs][:num_results]
        if resolve_links:
            results = await resolve_google_news_links(results, client)
        return results

    if http_client is not None:
        return await _run(http_client)
    async with httpx.AsyncClient(timeout=30) as client:
        return await _run(client)
