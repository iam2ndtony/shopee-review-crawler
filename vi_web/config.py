"""Web demo settings. Paths come from env vars so the same code runs from any checkout.

SAVED_REVIEWS_DIR: folder of crawled review JSON files (default <repo>/raw_reviews).
SHOPEE_PROFILE_DIR: Chrome profile folder with the saved Shopee login (default <repo>/shopee_chrome_profile).
Chrome allows one session per profile, so the web crawl and a manual crawl cannot run together.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from vi_clean.config import ROOT

MAX_CAP = 3000  # hard cap on reviews per live crawl


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name) or default)


@dataclass(frozen=True)
class WebConfig:
    saved_dir: Path = field(default_factory=lambda: _env_path("SAVED_REVIEWS_DIR", ROOT / "raw_reviews"))
    profile_dir: Path = field(default_factory=lambda: _env_path("SHOPEE_PROFILE_DIR", ROOT / "shopee_chrome_profile"))
    default_max: int = 1000
    max_cap: int = MAX_CAP
    ready_timeout_s: float = 600  # a job not marked ready within this ends as an error
    max_jobs: int = 5
    page_limit: int = 50  # reviews per ratings request
    delay_s: float = 1.0  # pause between requests, as the root crawler does
    retries: int = 3
