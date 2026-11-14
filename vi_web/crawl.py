"""Shopee review crawl for the web demo.

Written separately from the root shopee_crawler_persistent.py (which waits on stdin and has the
author's uncommitted edits): same profile folder, same ratings endpoint, same pacing. The page's
"I'm ready" button replaces the terminal ENTER. Only the fields the analysis needs are kept;
reviewer username, user id and avatar are dropped here and never leave this module.
No retry past a refusal, no captcha solving: a refusal ends the job with a plain message.
Scraping Shopee goes against its terms of service; this is for a few products on one machine.
"""
import itertools
import logging
import re
import threading
import time
from typing import Callable
from urllib.parse import urlparse

from .config import WebConfig

log = logging.getLogger(__name__)
_HOSTS = {"shopee.vn", "www.shopee.vn"}
_PATHS = (re.compile(r"-i\.(\d+)\.(\d+)$"), re.compile(r"^/product/(\d+)/(\d+)/?$"))
_FETCH_JS = """async (url) => {
    try {
        const res = await fetch(url);
        if (!res.ok) return { _fetch_error: `HTTP ${res.status}` };
        return await res.json();
    } catch (e) {
        return { _fetch_error: e.message };
    }
}"""


class CrawlError(Exception):
    """A crawl failure with a short code (refused, profile_in_use, no_playwright, timeout, bad_response)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def parse_product_url(url: str) -> tuple[str, str]:
    """(shop_id, item_id) from an https shopee.vn product link; ValueError for anything else."""
    p = urlparse(url.strip())
    if p.scheme != "https" or p.hostname not in _HOSTS or p.username or p.password or p.port not in (None, 443):
        raise ValueError("Not a Shopee product link (expected https://shopee.vn/...)")
    for pat in _PATHS:
        m = pat.search(p.path)
        if m:
            return m.group(1), m.group(2)
    raise ValueError("The link has no shop and item id (expected ...-i.<shop>.<item>)")


def keep_fields(item: dict) -> dict:
    """Only what the analysis reads; everything else, reviewer identity included, is dropped."""
    pi = item.get("product_items") or []
    return {
        "comment": item.get("comment"), "rating_star": item.get("rating_star"), "ctime": item.get("ctime"),
        "shopid": item.get("shopid"), "itemid": item.get("itemid"),
        "product_items": [{"model_name": pi[0].get("model_name")}] if pi and isinstance(pi[0], dict) else [],
    }


def _has_text(item) -> bool:
    """True for a review item with a non-blank comment. Shopee's ratings endpoint has no filter we could
    verify for this, so stars-only reviews are fetched with their page and dropped here."""
    return isinstance(item, dict) and isinstance(item.get("comment"), str) and bool(item["comment"].strip())


def wait_ready(ready: threading.Event, timeout_s: float) -> None:
    if not ready.wait(timeout_s):
        raise CrawlError("timeout", "Not marked ready in time. Start the analysis again.")


def fetch_all(fetch_page: Callable[[int], dict], max_reviews: int | None, cfg: WebConfig,
              on_progress: Callable[[int], None]) -> list[dict]:
    """Page through the ratings endpoint (cfg.page_limit per request) until Shopee returns an empty
    page; with max_reviews, stop there instead. max_reviews None means every review."""
    items: list[dict] = []
    offsets = itertools.count(0, cfg.page_limit) if max_reviews is None else range(0, max_reviews, cfg.page_limit)
    prev_first = None
    for offset in offsets:
        resp = None
        for attempt in range(1, cfg.retries + 1):
            try:
                resp = fetch_page(offset)
                break
            except Exception as e:  # the page may still be settling after login
                log.warning("ratings fetch failed at offset %s, attempt %s: %s", offset, attempt, type(e).__name__)
                if attempt == cfg.retries:
                    raise CrawlError("bad_response", "Could not read reviews from Shopee. Try again.") from e
                time.sleep(cfg.delay_s)
        if not isinstance(resp, dict):
            raise CrawlError("bad_response", "Shopee sent an unexpected answer.")
        if "_fetch_error" in resp:
            raise CrawlError("refused", f"Shopee refused the request ({resp['_fetch_error']}). "
                                        "Log in in the window that opened and try again.")
        if resp.get("error") not in (0, None):
            raise CrawlError("refused", f"Shopee returned error code {resp.get('error')}. "
                                        "Log in in the window that opened and try again.")
        ratings = (resp.get("data") or {}).get("ratings") or []
        if not ratings:
            break
        first = ratings[0].get("cmtid") if isinstance(ratings[0], dict) else None  # compared only, never kept
        if first is not None and first == prev_first:  # the endpoint ignored the offset: no new reviews, stop
            break
        prev_first = first
        items.extend(keep_fields(r) for r in ratings if _has_text(r))  # stars-only reviews are skipped here
        on_progress(len(items) if max_reviews is None else min(len(items), max_reviews))
        if max_reviews is not None and len(items) >= max_reviews:
            break
        time.sleep(cfg.delay_s)
    return items[:max_reviews]


def crawl(shop_id: str, item_id: str, max_reviews: int | None, cfg: WebConfig, ready: threading.Event,
          on_state: Callable[[str], None], on_progress: Callable[[int], None]) -> list[dict]:
    """Open Chrome with the saved login, wait for "ready", then fetch the reviews."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise CrawlError("no_playwright", "Playwright is not installed (pip install playwright, "
                                          "then playwright install chromium).") from e
    cfg.profile_dir.mkdir(parents=True, exist_ok=True)
    product_url = f"https://shopee.vn/product/{shop_id}/{item_id}"
    with sync_playwright() as p:
        try:
            ctx = p.chromium.launch_persistent_context(
                user_data_dir=str(cfg.profile_dir), headless=False, viewport={"width": 1280, "height": 800},
                ignore_default_args=["--enable-automation"],
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"])
        except Exception as e:
            raise CrawlError("profile_in_use", "Could not open the browser window. Close any other crawler or "
                                               "Chrome using the Shopee profile, or run playwright install chromium.") from e
        try:
            page = ctx.new_page()
            try:
                page.goto(product_url, wait_until="domcontentloaded", timeout=60000)
            except Exception as e:
                raise CrawlError("bad_response", "Could not open the Shopee product page.") from e
            on_state("waiting_login")
            wait_ready(ready, cfg.ready_timeout_s)
            on_state("crawling")
            try:  # re-open after login, as the root crawler does, so the login redirect has finished
                page.goto(product_url, wait_until="domcontentloaded", timeout=60000)
                time.sleep(2)
            except Exception:
                pass

            def fetch_page(offset: int) -> dict:
                url = (f"https://shopee.vn/api/v2/item/get_ratings?filter=0&limit={cfg.page_limit}"
                       f"&offset={offset}&itemid={item_id}&shopid={shop_id}&type=0")
                return page.evaluate(_FETCH_JS, url)

            return fetch_all(fetch_page, max_reviews, cfg, on_progress)
        finally:
            ctx.close()
