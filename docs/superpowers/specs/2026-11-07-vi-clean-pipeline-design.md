# Vietnamese Review Cleaning Pipeline: Design

Status: written from the in-chat design; the user has not yet confirmed it line by line.

## Purpose
First component of an end-to-end Vietnamese 3-class sentiment system. Turns raw reviews (UIT-VSFC + Shopee crawler output) into one clean, deduplicated dataset, leak-free by construction, so later work (Gold Set, split, augmentation, training, ablation) starts from a trusted file. Out of scope this iteration: Gold Test Set, splitting, EDA augmentation, classifier, baselines, evaluation, t-SNE, demo UI, LaTeX report, tests.

## Decisions
- New package `vi_clean/` in this repo. `shopee_crawler_persistent.py` and `clean_reviews.py` stay untouched. Crawl4AI is not used; the existing Playwright crawler is the Shopee source (`raw_reviews/*.json`).
- Stack: Python, NumPy, Pandas; FastAPI + uvicorn for the API. No other dependencies (no pyarrow, no tokenizer).
- Stages, one named function each, driven by an ordered config so they can be ablated: `nfc_normalize`, `regex_clean`, `expand_teencode` (per-text), then `drop_md5_duplicates`, `drop_low_ttr` (dataset-level).
- Order deviates from the brief's list (regex, NFC, ..., teencode last): NFC first so decomposed Vietnamese cannot defeat later regexes; teencode before dedup/TTR so "ko" and "không" count as the same text. The order is a config value.
- Defaults (all named in `CleanConfig`): lowercase on; character runs collapsed to max 2; emoji kept; MD5 key = casefolded text with punctuation removed and whitespace collapsed, keep first occurrence; TTR flag when tokens >= 5 and unique/total < 0.3.
- Teencode: `vi_clean/data/teencode_v1.json` = `{"version": "1.0.0", "entries": {...}}`, hand-authored (not derived from data), whole-token match only, case-insensitive. English commercial terms ("ship", "shop") are not rewritten.
- Labels: UIT-VSFC 0/1/2 = negative/neutral/positive. Shopee stars 1-2 negative, 3 neutral, 4-5 positive (weak labels; the `rating` column is kept).
- Merged schema: `text, label, source, rating` (rating NaN for UIT-VSFC). Shopee personal fields (username, userid) are never loaded.
- Merge job writes `data/clean/dataset.csv` (UTF-8) and `data/clean/report.json`: rows in per source, rows dropped per stage, same-text/different-label conflicts, final count per source and label, final count vs the brief's 34,250 with the explicit difference, and a note that Fleiss' kappa cannot be reproduced (no Gold Set exists). No filter is tuned to hit 34,250.
- UIT-VSFC is read in its original layout (`<split>/sents.txt`, `<split>/sentiments.txt`) from `data/uit_vsfc/`; train/dev/test are pooled, never split.
- API (`vi_clean/api.py`): `POST /clean {text}`, `POST /clean/batch {texts}` (cap 1000) return cleaned text and a `low_ttr` flag; `POST /jobs/merge-clean` (no body, configured paths only, synchronous) returns the report.

## Leak-free by construction
Dedup runs before any split; nothing is split; no dictionary, threshold or statistic is derived from the data.

## Known gaps to report, not hide
- Shopee data on disk: 1 product, 1,000 reviews (960 with text), 22 three-star. The brief expects 18,075 scraped reviews and 7.4% Neutral; neither is reproducible from this data.
- 34,250 is likely unreproducible; the report states the real count and gap.
