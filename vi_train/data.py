"""Data preparation and leak-free splits for fine-tuning.

All splits happen after the cleaning job's dedup. UIT-VSFC gets a stratified 80/10/10 split
(own split, not comparable with published UIT-VSFC results). Shopee is split by whole product
(leave-one-product-out) so reviews of one product never sit on both sides. Class weights
come from the training rows of a stage only.
"""
import dataclasses

import numpy as np
import pandas as pd
from underthesea import word_tokenize

from vi_clean.config import CleanConfig
from vi_clean.merge import run_job

from .config import TrainConfig


def segment(text: str) -> str:
    """Word-segment for PhoBERT ('sản phẩm' -> 'sản_phẩm'); empty in, empty out."""
    return word_tokenize(text, format="text") if text.strip() else ""


def prepare(cfg: TrainConfig) -> pd.DataFrame:
    """Run the cleaning job (lowercase off), add segmented text `seg` and label id `y`."""
    clean_cfg = dataclasses.replace(
        CleanConfig(), lowercase=False, shopee_dir=cfg.shopee_dir, out_dir=cfg.work_dir / "clean")
    run_job(clean_cfg)
    # keep_default_na=False: texts like "nan" or "null" must stay text, not become NaN.
    df = pd.read_csv(clean_cfg.out_dir / "dataset.csv", encoding="utf-8-sig", keep_default_na=False,
                     dtype={"text": str, "product_id": str})
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["product_id"] = df["product_id"].replace("", np.nan)

    cache_path = cfg.work_dir / "segmented.csv"
    cache = {}
    if cache_path.is_file():
        old = pd.read_csv(cache_path, encoding="utf-8-sig", keep_default_na=False, dtype=str)
        cache = dict(zip(old["text"], old["seg"]))
    missing = [t for t in df["text"].unique() if t not in cache]
    for t in missing:
        cache[t] = segment(t)
    if missing:
        pd.DataFrame({"text": list(cache), "seg": list(cache.values())}).to_csv(
            cache_path, index=False, encoding="utf-8-sig")
    df["seg"] = df["text"].map(cache)
    df["y"] = df["label"].map({lab: i for i, lab in enumerate(cfg.labels)})
    return df


def stratified_split(df: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (rest, held). Per label, held gets round(frac*n) rows, at least 1 when n >= 2
    and never all of them, so a tiny class always keeps a row in `rest`."""
    rng = np.random.default_rng(seed)
    held_idx: list = []
    for _, g in df.sort_index().groupby("label"):
        n = len(g)
        k = 0 if n < 2 else min(max(1, round(frac * n)), n - 1)
        held_idx.extend(rng.permutation(g.index.to_numpy())[:k])
    held = df.loc[sorted(held_idx)]
    return df.drop(index=held.index), held


def split_uit(df: pd.DataFrame, cfg: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """UIT-VSFC train / val / test, 80/10/10 stratified by label."""
    uit = df[df["source"] == "uit_vsfc"]
    train, held = stratified_split(uit, 0.2, cfg.seed)
    val, test = stratified_split(held, 0.5, cfg.seed + 1)
    return train, val, test


def shopee_folds(df: pd.DataFrame, cfg: TrainConfig):
    """One fold per product: (held-out product_id, train, val, test); val is a stratified
    val_frac of the training products' rows."""
    shop = df[df["source"] == "shopee"]
    folds = []
    for pid in sorted(shop["product_id"].unique()):
        test = shop[shop["product_id"] == pid]
        train, val = stratified_split(shop[shop["product_id"] != pid], cfg.val_frac, cfg.seed)
        folds.append((pid, train, val, test))
    return folds


def shopee_final(df: pd.DataFrame, cfg: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All Shopee rows split into train / val for the shipped model."""
    return stratified_split(df[df["source"] == "shopee"], cfg.val_frac, cfg.seed)


def class_weights(y: np.ndarray, k: int) -> np.ndarray:
    """w_c = N / (k * n_c); a class with no rows is an error, not a silent infinity."""
    counts = np.bincount(y, minlength=k)
    if (counts == 0).any():
        raise ValueError(f"class index {int(np.argmin(counts))} has no training rows")
    return len(y) / (k * counts)
