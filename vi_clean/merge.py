"""Merge UIT-VSFC and Shopee reviews into one schema and run the cleaning pipeline.

Output schema: text, label, source, rating (rating is NaN for UIT-VSFC).
The job only reports counts against the brief's expected total; it never tunes a
filter to reach it. Dedup runs on the pooled data and nothing is split here.
Run: python -m vi_clean.merge
"""
import json
import os

import numpy as np
import pandas as pd

from .config import CleanConfig
from .stages import clean_texts, drop_low_ttr, drop_md5_duplicates
from .teencode import load_teencode

COLUMNS = ["text", "label", "source", "rating"]


def load_uit_vsfc(cfg: CleanConfig) -> pd.DataFrame:
    """Read the original layout <uit_dir>/<split>/sents.txt + sentiments.txt, pooling all splits."""
    if not cfg.uit_dir.is_dir():
        raise FileNotFoundError(f"UIT-VSFC folder not found: {cfg.uit_dir}")
    frames = []
    for split in sorted(p for p in cfg.uit_dir.iterdir() if (p / "sents.txt").is_file()):
        with open(split / "sents.txt", encoding="utf-8") as f:
            texts = f.read().splitlines()
        with open(split / "sentiments.txt", encoding="utf-8") as f:
            labels = [cfg.uit_label_map[int(x)] for x in f.read().split()]
        if len(texts) != len(labels):
            raise ValueError(f"{split}: {len(texts)} sentences but {len(labels)} sentiments")
        frames.append(pd.DataFrame({"text": texts, "label": labels, "source": "uit_vsfc", "rating": np.nan}))
    if not frames:
        raise FileNotFoundError(f"No <split>/sents.txt found under {cfg.uit_dir}")
    return pd.concat(frames, ignore_index=True)[COLUMNS]


def load_shopee(cfg: CleanConfig) -> pd.DataFrame:
    """Read every crawler JSON; keep only comment and star (no usernames or ids)."""
    files = sorted(cfg.shopee_dir.glob("*.json")) if cfg.shopee_dir.is_dir() else []
    if not files:
        raise FileNotFoundError(f"No Shopee JSON files found in {cfg.shopee_dir}")
    rows = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            for item in json.load(f):
                label = cfg.star_to_label.get(item.get("rating_star"))
                if label is not None:  # missing / unmapped star = no label, skip
                    rows.append((item.get("comment") or "", label, "shopee", float(item["rating_star"])))
    return pd.DataFrame(rows, columns=COLUMNS)


def run_job(cfg: CleanConfig = CleanConfig()) -> dict:
    """Merge, clean, dedup, TTR-filter, write dataset.csv + report.json, return the report."""
    table = load_teencode(cfg.teencode_path)
    df = pd.concat([load_uit_vsfc(cfg), load_shopee(cfg)], ignore_index=True)
    rows_in = df["source"].value_counts().to_dict()

    df["text"] = clean_texts(df["text"], cfg, table)
    n0 = len(df)
    df = df[df["text"] != ""]
    n_empty = n0 - len(df)
    df, conflicts = drop_md5_duplicates(df, cfg)
    n_md5 = n0 - n_empty - len(df)
    n1 = len(df)
    df = drop_low_ttr(df, cfg)
    n_ttr = n1 - len(df)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    tmp = cfg.out_dir / "dataset.csv.tmp"  # write then replace: readers never see a partial file
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, cfg.out_dir / "dataset.csv")

    total = len(df)
    labels = df["label"].value_counts().to_dict()
    neutral = labels.get("neutral", 0) / total if total else 0.0
    report = {
        "rows_in": rows_in,
        "dropped": {"empty": n_empty, "md5_duplicate": n_md5, "low_ttr": n_ttr},
        "label_conflicts": conflicts,
        "final": {"total": total, "per_source": df["source"].value_counts().to_dict(), "per_label": labels},
        "expected_total": cfg.expected_total,
        "difference": total - cfg.expected_total,
        "notes": [
            f"Neutral share is {neutral:.1%}; the brief states 7.4%.",
            "Fleiss' kappa cannot be reproduced: no Gold Test Set exists in this iteration.",
        ],
    }
    tmp = cfg.out_dir / "report.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=int)
    os.replace(tmp, cfg.out_dir / "report.json")
    return report


if __name__ == "__main__":
    print(json.dumps(run_job(), indent=2, ensure_ascii=False, default=int))
