# Cleaning pipeline: figures not reproduced and defaults chosen

Generated from the real run on 2026-11-07 (`python -m vi_clean.merge`, `data/clean/report.json`).

## Brief figures versus what the data gives

| Brief | This run | Status |
|---|---|---|
| UIT-VSFC 16,175 samples | 16,175 (11,426 train + 1,583 validation + 3,166 test) | Reproduced |
| Crawled reviews 18,075 | 1,000 (one product, 960 with text) | Not reproduced: only one product has been crawled |
| Clean total 34,250 | 16,865 (UIT-VSFC 16,001 + Shopee 864) | Not reproduced; difference -17,385. No filter was tuned. |
| Neutral 7.4% | 4.2% (703 of 16,865) | Not reproduced; Shopee is 22 three-star reviews of 1,000 |
| Gold Test Set 1,500 and Fleiss' kappa | Not built | Out of scope this iteration; kappa cannot be computed |
| Neutral 4,982 after EDA | Not built | Augmentation out of scope this iteration |

Drops in this run: 40 empty after cleaning, 270 MD5 exact duplicates, 0 low-TTR. Six dropped duplicates had a different label from the kept row (the first row is kept).

## Defaults chosen where the brief was open (all in `vi_clean/config.py`)

- Stage order: NFC, regex, teencode (per text), then MD5 dedup, then TTR. The brief lists regex first and teencode last; NFC runs first so decomposed Vietnamese cannot defeat the regexes, and teencode runs before dedup so "ko" and "không" match.
- Lowercase on; character runs capped at 2 (digits untouched); emoji kept; tone marks never stripped.
- MD5 key: casefolded text, punctuation removed, whitespace collapsed; first occurrence kept; exact match only.
- TTR: whitespace tokens; flagged when at least 5 tokens and unique/total below 0.3.
- Labels: UIT-VSFC 0/1/2 = negative/neutral/positive. Shopee 1-2 stars negative, 3 neutral, 4-5 positive (weak labels; `rating` column kept).
- Teencode: `vi_clean/data/teencode_v1.json`, hand-authored, whole-token, case-insensitive.

## Data provenance

UIT-VSFC came from the Hugging Face parquet conversion of `uitnlp/vietnamese_students_feedback` (the dataset card lists the license as unknown; check the original UIT NLP source before redistributing). It was converted once to `data/uit_vsfc/<split>/sents.txt` and `sentiments.txt` using a throwaway script; `pyarrow` is not a project dependency. `data/` is git-ignored.
