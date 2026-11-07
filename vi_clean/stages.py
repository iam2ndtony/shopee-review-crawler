"""The five cleaning stages, one named function each so they can be ablated.

Per-text stages (Series in, Series out): nfc_normalize, regex_clean, expand_teencode,
composed by clean_texts in cfg.stage_order. Dataset-level stages (DataFrame in, out):
drop_md5_duplicates, drop_low_ttr. Tone marks are never stripped.

Choices (all parameters live in CleanConfig):
- MD5 key: casefolded text, punctuation removed, whitespace collapsed; first row kept.
  Exact-match only; near-duplicates are out of scope.
- TTR: whitespace tokens; flagged when >= min_tokens_for_ttr tokens and unique/total < ttr_min.
"""
import hashlib
import re
import unicodedata

import pandas as pd

from .config import CleanConfig
from .teencode import compile_teencode

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_PHONE = re.compile(r"(?<!\d)(?:\+?84|0)[\d .-]{8,12}(?!\d)")
_HTML = re.compile(r"<[^>]+>")
# Shopee review template prefixes (same list as clean_reviews.py).
_TEMPLATE = re.compile(
    r"(Chất lượng sản phẩm|Tính năng nổi bật|Đúng với mô tả|Chất liệu|Thiết kế|Công dụng|Mùi hương|Độ bền):\s*",
    re.IGNORECASE)
_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def nfc_normalize(s: pd.Series) -> pd.Series:
    """Compose to Unicode NFC so 'e'+U+0301 equals 'é'."""
    return s.map(lambda x: unicodedata.normalize("NFC", x))


def regex_clean(s: pd.Series, cfg: CleanConfig) -> pd.Series:
    """Drop URLs, emails, phones, HTML, Shopee template prefixes; squeeze repeats and spaces."""
    repeat = re.compile(r"(\D)\1{%d,}" % cfg.max_char_repeat)  # digits left alone
    keep = r"\1" * cfg.max_char_repeat

    def one(t: str) -> str:
        for p in (_URL, _EMAIL, _HTML, _TEMPLATE, _PHONE):
            t = p.sub(" ", t)
        t = repeat.sub(keep, t)
        t = _SPACE.sub(" ", t).strip()
        return t.lower() if cfg.lowercase else t

    return s.map(one)


def expand_teencode(s: pd.Series, table: dict[str, str]) -> pd.Series:
    """Replace whole-token teencode using the loaded dictionary."""
    return s.map(compile_teencode(table))


def clean_texts(s: pd.Series, cfg: CleanConfig, table: dict[str, str]) -> pd.Series:
    """Run the per-text stages in cfg.stage_order; null / non-string values become ''."""
    out = s.map(lambda x: x if isinstance(x, str) else "")
    steps = {
        "nfc": nfc_normalize,
        "regex": lambda x: regex_clean(x, cfg),
        "teencode": lambda x: expand_teencode(x, table),
    }
    for name in cfg.stage_order:
        out = steps[name](out)
    return out.map(lambda t: _SPACE.sub(" ", t).strip())


def _md5_key(text: str) -> str:
    norm = _SPACE.sub(" ", _PUNCT.sub("", text.casefold())).strip()
    return hashlib.md5(norm.encode("utf-8")).hexdigest() if norm else ""


def drop_md5_duplicates(df: pd.DataFrame, cfg: CleanConfig) -> tuple[pd.DataFrame, int]:
    """Keep the first row per MD5 key; also return how many dropped rows had a different label."""
    key = df["text"].map(_md5_key)
    dup = key.duplicated(keep="first") & (key != "")
    first_label = df["label"].groupby(key).transform("first")
    conflicts = int((dup & (df["label"] != first_label)).sum())
    return df[~dup], conflicts


def low_ttr_mask(s: pd.Series, cfg: CleanConfig) -> pd.Series:
    """True where a text is long enough to judge and too repetitive (bot-like)."""
    def flag(t: str) -> bool:
        tokens = t.split()
        n = len(tokens)
        return n >= cfg.min_tokens_for_ttr and len(set(tokens)) / n < cfg.ttr_min

    return s.map(flag)


def drop_low_ttr(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    return df[~low_ttr_mask(df["text"], cfg)]
