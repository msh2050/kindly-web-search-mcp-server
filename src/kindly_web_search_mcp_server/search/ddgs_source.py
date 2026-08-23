"""ddgs as a supplementary search source.

`ddgs <https://github.com/deedy5/ddgs>`_ ("Dux Distributed Global Search", MIT)
is a metasearch *library* -- no server, no browser, no API key. It reaches
several engines through its own request shaping, and on some networks that
succeeds where a SearXNG instance's scrapers do not.

Measured against a local SearXNG (2026-08-24, ddgs 9.14.4 -- note that 9.15.0
no longer ships the bing or yandex backends):

===========  ==========  ====================
engine       ddgs        SearXNG
===========  ==========  ====================
bing         5 results   TCP unreachable
brave        5 results   HTTP 429
yahoo        5 results   HTTP 500
yandex       5 results   works
duckduckgo   no results  works
===========  ==========  ====================

The two are complementary rather than redundant, which is the whole reason to
run both: `bing` in particular answers through ddgs even though a direct
connection to ``bing.com`` from here times out.

``ddgs`` is an optional import. When it is missing this module returns nothing
and says so once, so the provider degrades to its other sources instead of
failing.
"""

from __future__ import annotations

import asyncio
import logging
import os

from ..models import WebSearchResult

LOGGER = logging.getLogger(__name__)

# "auto" rather than a pinned list. Naming the engines measured to work here
# was ~1.2s faster, but the set is not stable: 9.14.4 offered bing and yandex,
# 9.15.0 dropped both, and a pinned name that no longer exists is rejected with
# "backends do not exist or are disabled". Letting ddgs pick survives its own
# churn; pin with KINDLY_DDGS_BACKENDS if a specific engine matters.
DEFAULT_BACKENDS = "auto"

_missing_logged = False


def backends() -> str:
    return (os.environ.get("KINDLY_DDGS_BACKENDS") or "").strip() or DEFAULT_BACKENDS


def is_enabled() -> bool:
    """ddgs is on unless explicitly disabled."""
    raw = (os.environ.get("KINDLY_DDGS") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _search_blocking(query: str, limit: int, timelimit: str | None) -> list[dict]:
    """Run the (synchronous) ddgs query.

    ddgs signals an empty sweep by *raising* rather than returning ``[]``, and
    a refusal from one engine looks the same as a genuinely empty result, so
    everything is funnelled into an empty list for the caller.
    """
    global _missing_logged
    try:
        from ddgs import DDGS
    except ImportError:
        if not _missing_logged:
            LOGGER.info("ddgs is not installed; skipping that source "
                        "(pip install ddgs to enable it)")
            _missing_logged = True
        return []

    kwargs = {"max_results": limit, "backend": backends()}
    if timelimit:
        kwargs["timelimit"] = timelimit
    try:
        return DDGS(timeout=20).text(query, **kwargs) or []
    except Exception as exc:  # noqa: BLE001 -- deliberate
        # ddgs raises for an empty sweep, a rate limit and a refused engine
        # alike, and the exception types are not part of its public API.
        # None of them should fail the search: the other sources still have
        # results, so this degrades to contributing nothing.
        LOGGER.info("ddgs returned nothing (%s: %s)", type(exc).__name__,
                    str(exc)[:80])
        return []


async def search_ddgs(
    query: str,
    *,
    num_results: int,
    timelimit: str | None = None,
) -> list[WebSearchResult]:
    """Query ddgs for ``query``.

    Args:
        query: The search query.
        num_results: Maximum number of results.
        timelimit: Optional recency window -- ``d``, ``w``, ``m`` or ``y``.

    Returns:
        Up to ``num_results`` results; an empty list on any failure.
    """
    if not query.strip() or num_results < 1 or not is_enabled():
        return []

    # DDGS.text() is blocking, so it cannot run on the event loop: doing so
    # would stall every other source this provider queries in parallel.
    rows = await asyncio.to_thread(_search_blocking, query, num_results, timelimit)

    out: list[WebSearchResult] = []
    for row in rows:
        title = (row.get("title") or "").strip()
        link = (row.get("href") or "").strip()
        body = (row.get("body") or "").strip()
        if not title or not link.startswith(("http://", "https://")):
            continue
        out.append(
            WebSearchResult(title=title, link=link, snippet=body or title,
                            page_content="")
        )
        if len(out) >= num_results:
            break
    return out
