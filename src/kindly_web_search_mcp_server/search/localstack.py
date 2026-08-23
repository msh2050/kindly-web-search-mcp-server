"""A composite provider: SearXNG + Google News RSS, reranked and filtered.

:mod:`..search` selects exactly one provider by strict priority and never falls
back or merges, which is the right default but leaves no way to combine sources.
This provider is that combination, expressed as a single provider so the router
does not have to learn about merging.

What it adds over calling SearXNG directly:

* **Google News RSS merged in.** The only reliably reachable route to Google's
  news index on networks where the scraper-based engines are blocked.
* **Query-overlap reranking.** SearXNG mixes engine ranks across very different
  engines and does not AND the query terms, so a result matching only filler
  words can outrank an on-topic one -- "Iraq news today" surfacing a football
  piece titled "transfer news and rumours today" is a real observed case.
* **ddgs merged in.** A metasearch library that reaches bing, brave and yahoo
  through its own request shaping -- all three of which this SearXNG instance
  cannot use (unreachable, 429 and 500 respectively). Complementary coverage
  rather than duplicated coverage.
* **Ignored domains.** Some hosts cost a content-extraction slot and can never
  repay it: msn.com 301s every article to a localised homepage, and the social
  networks are login walls.

Everything here is additive; no upstream file is modified beyond one entry in
:data:`..search.PROVIDERS`, which keeps rebases cheap.

Enable with ``KINDLY_LOCAL_STACK=1`` alongside ``SEARXNG_BASE_URL``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from urllib.parse import urlparse

import httpx

from ..models import WebSearchResult
from .ddgs_source import search_ddgs
from .googlenews import resolve_google_news_links, search_google_news
from .searxng import search_searxng

LOGGER = logging.getLogger(__name__)

DEFAULT_IGNORED_DOMAINS = (
    "msn.com",            # 301s every article to a localised homepage
    "facebook.com",
    "web.facebook.com",
    "instagram.com",
    "x.com",
    "twitter.com",
    "tiktok.com",
    # Video pages yield no readable text, so one costs a content-extraction
    # slot and returns nothing. Observed taking a slot on a research query.
    "youtube.com",
    "youtu.be",
)

# Words that carry no topic signal. Without this, "Iraq news today" ranks a
# result titled "transfer news and rumours today" first, because the filler
# words alone can win when the engine does not AND the query terms.
_STOPWORDS = frozenset(
    ["a", "an", "the", "of", "for", "in", "on", "at", "to", "from", "by", "with", "about", "into", "over", "after", "and", "or", "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "it", "its", "as", "new", "news", "latest", "today", "todays", "update", "updates", "recent", "current", "breaking", "report", "reports", "info", "information", "please", "search", "find", "show", "tell", "me", "my", "what", "when", "where", "which", "who", "why", "how", "do", "does", "did"]
)

# Terms that make a query worth asking Google News about. A plain web query
# gains nothing from a news feed, and the extra request is pure latency.
_NEWS_HINTS = frozenset(
    ["news", "headline", "headlines", "breaking", "latest", "today", "yesterday", "update", "updates", "announced", "announcement", "report", "reports", "live", "coverage"]
)

_WORD_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)

# Source quality, used only to break ties between equally on-topic results.
# Retrieval was never the weak part of a research query -- ranking was: a query
# where every result mentions the topic leaves the overlap score flat, and
# whatever the engines happened to return first wins. A vendor blog and a
# standards body should not be interchangeable in that position.
#
# Suffix match, so "nist.gov" covers "www.nist.gov" and ".gov" covers any
# government host.
AUTHORITY = (
    (3, (".gov", ".edu", ".mil", ".int", ".ac.uk", ".edu.au", ".ac.jp",
         "arxiv.org", "doi.org", "nature.com", "science.org", "sciencedirect.com",
         "ieee.org", "acm.org", "springer.com", "pubmed.ncbi.nlm.nih.gov",
         "nih.gov", "who.int", "europa.eu", "oecd.org", "imf.org",
         "worldbank.org", "un.org")),
    (2, ("wikipedia.org", "reuters.com", "apnews.com", "bbc.co.uk", "bbc.com",
         "ft.com", "economist.com", "nature.org", "scientificamerican.com",
         "arstechnica.com", "aljazeera.com", "bloomberg.com", "wsj.com",
         "github.com", "stackoverflow.com", "stackexchange.com",
         "python.org", "mozilla.org", "docs.rs", "readthedocs.io")),
    # Demoted: user-generated or aggregated pages that restate a source without
    # being one. Never removed -- for some queries they are the only coverage.
    (-1, ("linkedin.com", "quora.com", "pinterest.com", "blogspot.com",
          "wordpress.com", "medium.com", "substack.com", "reddit.com",
          "slideshare.net", "scribd.com")),
)


def authority_ranking_enabled() -> bool:
    return (os.environ.get("KINDLY_AUTHORITY_RANKING") or "").strip().lower() \
        not in ("0", "false", "no", "off")


def authority_score(url: str) -> int:
    """Rank a host by how far it is from being a primary source."""
    if not authority_ranking_enabled():
        return 0
    host = _host(url)
    if not host:
        return 0
    for score, suffixes in AUTHORITY:
        for suffix in suffixes:
            if host == suffix or host.endswith(suffix if suffix.startswith(".")
                                               else "." + suffix):
                return score
    return 0


def _flag(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def is_enabled() -> bool:
    """Report whether this provider is switched on.

    Deliberately "any non-blank value enables it", matching how upstream selects
    every other provider (a `SERPER_API_KEY` of "0" selects Serper too). Unset
    the variable to turn it off; do not set it to 0.
    """
    return bool((os.environ.get("KINDLY_LOCAL_STACK") or "").strip())


def ignored_domains() -> tuple[str, ...]:
    raw = (os.environ.get("KINDLY_IGNORED_DOMAINS") or "").strip()
    if not raw:
        return DEFAULT_IGNORED_DOMAINS
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def drop_ignored(results: list[WebSearchResult]) -> list[WebSearchResult]:
    """Remove results whose host is on the ignore list.

    Done before the extraction stage so a dead host never consumes one of the
    few slots the caller asked for.
    """
    blocked = ignored_domains()
    if not blocked:
        return results
    kept = []
    for result in results:
        host = _host(result.link)
        if host and any(host == d or host.endswith("." + d) for d in blocked):
            continue
        kept.append(result)
    return kept


def query_terms(query: str) -> set[str]:
    return {w for w in _WORD_RE.findall((query or "").lower())} - _STOPWORDS


def rerank(results: list[WebSearchResult], query: str) -> list[WebSearchResult]:
    """Sort by topic-word overlap, breaking ties on source authority.

    Overlap stays the primary key, so a more on-topic page always outranks a
    more authoritative one -- authority only decides between results that match
    the query equally well. In practice that is most of them: ask about quantum
    computing and every result says "quantum", leaving the order to whichever
    engine answered first. Original order is the final tiebreak, so this never
    invents a ranking where it has no signal.
    """
    terms = query_terms(query)
    if not terms and not authority_ranking_enabled():
        return results

    def overlap(result: WebSearchResult) -> int:
        haystack = f"{result.title} {result.snippet}".lower()
        return sum(1 for term in terms if term in haystack)

    def score(result: WebSearchResult) -> int:
        # Overlap is doubled so authority is bounded: it can overturn a
        # one-word difference in topical match but never a larger one. As a
        # pure tiebreak it would almost never fire -- results rarely match a
        # query *equally* -- and a blog repeating one extra query word would
        # keep outranking a standards body, which is the whole problem here.
        return overlap(result) * 2 + authority_score(result.link)

    ordered = sorted(
        enumerate(results), key=lambda pair: (score(pair[1]), -pair[0]), reverse=True
    )
    return [result for _, result in ordered]


def _ddgs_timelimit() -> str | None:
    """Recency window handed to ddgs for news-shaped queries.

    ``d``/``w``/``m``/``y``; blank disables it. A week is wide enough to
    survive a quiet news day and narrow enough to exclude archive pages.
    """
    raw = (os.environ.get("KINDLY_DDGS_NEWS_TIMELIMIT") or "w").strip().lower()
    return raw if raw in ("d", "w", "m", "y") else None


def looks_like_news(query: str) -> bool:
    """Guess whether a query wants current events."""
    if _flag("KINDLY_GOOGLE_NEWS_ALWAYS"):
        return True
    words = {w for w in _WORD_RE.findall((query or "").lower())}
    return bool(words & _NEWS_HINTS)


def merge(
    primary: list[WebSearchResult], extra: list[WebSearchResult]
) -> list[WebSearchResult]:
    """Combine two result lists, dropping duplicates by URL then by title."""
    seen_links = {r.link.rstrip("/").lower() for r in primary}
    seen_titles = {r.title.strip().lower() for r in primary}
    merged = list(primary)
    for result in extra:
        link = result.link.rstrip("/").lower()
        title = result.title.strip().lower()
        if link in seen_links or title in seen_titles:
            continue
        seen_links.add(link)
        seen_titles.add(title)
        merged.append(result)
    return merged


async def search_local_stack(
    query: str,
    *,
    num_results: int,
    http_client: httpx.AsyncClient | None = None,
) -> list[WebSearchResult]:
    """Search SearXNG, optionally merge Google News, then rerank and filter.

    A Google News failure is logged and ignored: the SearXNG results are still a
    complete answer, and a feed outage should not fail the whole search.
    """
    if not query.strip() or num_results < 1:
        return []

    async def _run(client: httpx.AsyncClient) -> list[WebSearchResult]:
        want_news = looks_like_news(query)
        # Over-fetch from SearXNG so reranking and domain filtering have room to
        # discard without dropping below num_results.
        overfetch = max(num_results * 4, 20)

        # Named indices rather than positional ones: the news task is
        # conditional, so a positional read silently shifts when it is absent.
        tasks: dict = {
            "searxng": search_searxng(query, num_results=overfetch, http_client=client),
            # Recency window only when the query asks for it; ddgs would
            # otherwise happily return decade-old pages for "news today".
            "ddgs": search_ddgs(
                query,
                num_results=max(num_results * 2, 10),
                timelimit=_ddgs_timelimit() if want_news else None,
            ),
        }
        if want_news:
            tasks["news"] = search_google_news(
                query,
                num_results=max(num_results * 2, 10),
                http_client=client,
                resolve_links=False,   # resolve only the survivors, below
            )

        done = dict(zip(tasks, await asyncio.gather(*tasks.values(),
                                                    return_exceptions=True)))

        primary = done["searxng"]
        if isinstance(primary, BaseException):
            raise primary
        results: list[WebSearchResult] = list(primary)

        if want_news:
            news = done["news"]
            if isinstance(news, BaseException):
                LOGGER.warning("Google News unavailable (%s); using SearXNG only",
                               type(news).__name__)
            else:
                LOGGER.info("Google News contributed %d headline(s)", len(news))
                # News leads for a news query. Appending it instead buries it:
                # reranking scores topic-word overlap, so a query like "Iraq news
                # today" ties every "Iraq" result at one point, the stable sort
                # keeps the SearXNG entries in front, and the trim to num_results
                # drops the fresh headlines entirely -- observed returning site
                # hub pages and a 2010 article for a "today" query.
                results = merge(list(news), results)

        extra = done["ddgs"]
        if isinstance(extra, BaseException):
            LOGGER.info("ddgs unavailable (%s)", type(extra).__name__)
        elif extra:
            before = len(results)
            # Appended, not led: SearXNG returns an order of magnitude more
            # results, so ddgs is filling gaps rather than setting the order.
            # Reranking decides the final order anyway.
            results = merge(results, list(extra))
            LOGGER.info("ddgs added %d result(s) (%d new)",
                        len(extra), len(results) - before)

        results = drop_ignored(results)
        results = rerank(results, query)
        results = results[:num_results]
        # Wrapper links only become real URLs here, once the set is final.
        return await resolve_google_news_links(results, client)

    if http_client is not None:
        return await _run(http_client)
    async with httpx.AsyncClient(timeout=30) as client:
        return await _run(client)
