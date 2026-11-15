"""Review analysis for the web demo: rows, flags and the EDA numbers.

Only the review text, stars, date and variation are read from a crawled item; reviewer
username, user id and avatar are never touched. Stars-only reviews (no text) are counted but
not listed or predicted. A review is "not predicted" when it has text but no label.
"""
import dataclasses
import json
import re
from datetime import datetime, timezone

import pandas as pd

from vi_clean.config import CleanConfig
from vi_clean.predictor import predict_texts, serve_run
from vi_clean.stages import clean_texts, low_ttr_mask, md5_key, strip_coin_phrase
from vi_clean.teencode import load_teencode
from vi_train.config import TrainConfig

from .config import WebConfig

LABELS = TrainConfig().labels  # ("negative", "neutral", "positive")
# Training data was cleaned with lowercase off; the analysis cleans the same way.
_CFG = dataclasses.replace(CleanConfig(), lowercase=False)
_TABLE = load_teencode(_CFG.teencode_path)


def _date(ts) -> str:
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") if ts else ""
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


def reviews_frame(items: list[dict]) -> pd.DataFrame:
    """One row per crawled item: n (1-based), stars, date (UTC), variation, text ("" when stars-only)."""
    rows = []
    for n, it in enumerate(items, 1):
        pi = it.get("product_items") or []
        comment = it.get("comment")
        rows.append({
            "n": n,
            "stars": int(it.get("rating_star") or 0),
            "date": _date(it.get("ctime")),
            "variation": (pi[0].get("model_name") or "") if pi and isinstance(pi[0], dict) else "",
            "text": comment.strip() if isinstance(comment, str) else "",
        })
    return pd.DataFrame(rows, columns=["n", "stars", "date", "variation", "text"])


def product_id_of(items: list[dict]) -> str:
    """`s<shopid>_i<itemid>` of the first item, the id the training data uses; "" if missing."""
    if items and items[0].get("shopid") and items[0].get("itemid"):
        return f"s{items[0]['shopid']}_i{items[0]['itemid']}"
    return ""


def flag_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned text and the duplicate / repetitive flags (False for stars-only rows)."""
    out = df.copy()
    out["cleaned"] = clean_texts(out["text"], _CFG, _TABLE)
    keys = out["cleaned"].map(md5_key)
    out["duplicate"] = (keys.duplicated(keep="first") & (keys != "")).astype(bool)
    out["repetitive"] = low_ttr_mask(out["cleaned"], _CFG).astype(bool)
    return out


def list_saved(cfg: WebConfig) -> list[dict]:
    """Saved review files (JSON lists) in cfg.saved_dir, then the live crawls in its crawled/ folder."""
    out = []
    for sub, tag in (("", ""), ("crawled/", " (crawled)")):
        folder = cfg.saved_dir / sub
        for p in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, list):
                # crawled files are <name>_<shop>_<item>: show the name without the ids
                name = re.sub(r"_\d+_\d+$", "", p.stem).replace("-", " ") if sub else p.stem
                out.append({"file": sub + p.name, "name": name + tag, "reviews": len(data)})
    return out


def _load_saved(file: str, cfg: WebConfig) -> list[dict]:
    if file not in {p["file"] for p in list_saved(cfg)}:  # only names from the listing, never a path
        raise KeyError(file)
    return json.loads((cfg.saved_dir / file).read_text(encoding="utf-8"))


def _oof_path():
    return TrainConfig().work_dir / "preds" / serve_run() / "shopee_oof.csv"


def trained_product_ids() -> set[str]:
    """Product ids in the served run's held-out table: the products its model was trained on."""
    try:
        oof = pd.read_csv(_oof_path(), encoding="utf-8-sig", keep_default_na=False, dtype=str, usecols=["product_id"])
    except FileNotFoundError:
        return set()
    return set(oof["product_id"]) - {""}


def analyze_saved(file: str, cfg: WebConfig) -> dict:
    """Analysis of a saved product with the held-out predictions of leave-one-product-out.

    A review is matched to a prediction of the same product by the MD5 key of its cleaned text.
    The dataset was deduplicated across all products, so a review repeated in another product
    has no held-out prediction here; borrowing the other product's label would leak (that
    product's fold model trained on this one), so it stays "not predicted".
    """
    items = _load_saved(file, cfg)
    if file.startswith("crawled/"):  # a new product has no held-out predictions: the final model has not seen it
        return analyze_live(items, cfg)
    df, dropped, trimmed = _drop_coin(reviews_frame(items))
    df = flag_frame(df)
    tcfg = TrainConfig()
    oof = pd.read_csv(_oof_path(), encoding="utf-8-sig", keep_default_na=False, dtype=str)
    oof = oof[oof["product_id"] == product_id_of(items)]
    # `pred` is the model's held-out prediction; `y` is the true star-derived label and must not be shown as one.
    by_key = {md5_key(t): tcfg.labels[int(p)] for t, p in zip(oof["text"], oof["pred"])}
    # Match on the review as it was in the dataset, before the coin phrase was cut.
    raw_clean = clean_texts(df["raw"], _CFG, _TABLE)
    df["label"] = [by_key.get(md5_key(r)) if c else None for r, c in zip(raw_clean, df["cleaned"])]
    df["probs"] = None  # the saved predictions store labels only
    return build_analysis(df, "saved", False, dropped, trimmed)


def analyze_live(items: list[dict], cfg: WebConfig | None = None) -> dict:
    """Analysis of freshly crawled reviews with the final model; probabilities are included."""
    df, dropped, trimmed = _drop_coin(reviews_frame(items))
    df = flag_frame(df)
    df["label"], df["probs"] = None, None
    idx = df.index[df["text"] != ""].tolist()
    if idx:
        preds = predict_texts(df.loc[idx, "text"].tolist())
        df["label"] = df["label"].astype(object)
        df["probs"] = df["probs"].astype(object)
        for i, p in zip(idx, preds):
            df.at[i, "label"], df.at[i, "probs"] = p["label"], p["probs"]
    warn = product_id_of(items) in trained_product_ids()
    return build_analysis(df, "live", warn, dropped, trimmed)


def _disagrees(stars: int, label) -> bool:
    return (stars >= 4 and label == "negative") or (1 <= stars <= 2 and label == "positive")


def _count(rows: pd.DataFrame) -> dict:
    return {lab: int((rows["label"] == lab).sum()) for lab in LABELS}


def _drop_coin(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Cut the "nhan xu" phrase out of reviews; a review with nothing else left is removed.

    Returns the frame (with `raw`, the text before the cut), how many reviews were removed and how many trimmed.
    Done first, so the cut text is what gets cleaned, predicted and compared for duplicates."""
    cut = df["text"].map(lambda t: strip_coin_phrase(t, _CFG) if t else t)
    changed = cut != df["text"]
    gone = changed & (cut == "")
    out = df.assign(raw=df["text"], text=cut)
    return out[~gone].reset_index(drop=True), int(gone.sum()), int((changed & ~gone).sum())


def build_analysis(df: pd.DataFrame, mode: str, training_data_warning: bool, dropped: int = 0, trimmed: int = 0) -> dict:
    """The result JSON. `df` needs the flag_frame columns plus `label` (str or None) and `probs` (dict or None).
    `dropped` is how many reviews held only a coin phrase and were removed; `trimmed` how many had it cut out."""
    text_df = df[df["text"] != ""]
    pred = text_df[text_df["label"].notna()]
    n_pred = len(pred)
    counts = _count(pred)
    dates = df.loc[df["date"] != "", "date"]
    reviews = []
    for r in text_df.itertuples(index=False):
        label = r.label if isinstance(r.label, str) else None
        reviews.append({
            "n": int(r.n), "stars": int(r.stars), "date": r.date, "variation": r.variation,
            "text": r.text, "cleaned": r.cleaned, "label": label,
            "probs": r.probs if isinstance(r.probs, dict) else None,
            "duplicate": bool(r.duplicate), "repetitive": bool(r.repetitive),
            "disagrees": _disagrees(int(r.stars), label),
        })
    month = pred.loc[pred["date"] != ""].assign(month=lambda d: d["date"].str[:7])
    return {
        "summary": {
            "reviews_found": len(df), "with_text": len(text_df), "stars_only": len(df) - len(text_df),
            "predicted": n_pred, "not_predicted": len(text_df) - n_pred,
            "share": {lab: round(c / n_pred, 4) if n_pred else 0.0 for lab, c in counts.items()},
            "duplicates": int(text_df["duplicate"].sum()), "repetitive": int(text_df["repetitive"].sum()),
            "disagreements": sum(v["disagrees"] for v in reviews),
            "date_range": [dates.min(), dates.max()] if len(dates) else None,
            "mode": mode, "training_data_warning": training_data_warning, "coin_farming_dropped": dropped,
            "coin_phrase_trimmed": trimmed,
        },
        "stars_all": {str(s): int((df["stars"] == s).sum()) for s in range(1, 6)},
        "sentiment_counts": counts,
        "stars_by_label": {str(s): _count(pred[pred["stars"] == s]) for s in range(1, 6)},
        "by_month": [{"month": m, **_count(g)} for m, g in sorted(month.groupby("month"), key=lambda t: t[0])],
        "reviews": reviews,
    }
