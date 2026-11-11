# Cleaning pipeline: figures not reproduced and defaults chosen

Generated from the latest real run, on UIT-VSFC plus 5 crawled Shopee products (`vi_clean.merge` through `vi_train.prepare`, `data/train/clean/report.json`). Training runs use `lowercase=False`; the counts are identical with lowercase on.

## Brief figures versus what the data gives

| Brief | This run | Status |
|---|---|---|
| UIT-VSFC 16,175 samples | 16,175 (11,426 train + 1,583 validation + 3,166 test) | Reproduced |
| Crawled reviews 18,075 | 8,314 raw (5 Shopee products); 3,315 remain after cleaning | Not reproduced: only 5 products crawled, and about 55% of reviews have a star rating but no text |
| Clean total 34,250 | 19,316 (UIT-VSFC 16,001 + Shopee 3,315) | Not reproduced; difference -14,934. No filter was tuned. |
| Neutral 7.4% | 4.0% (773 of 19,316; Shopee has 92) | Not reproduced; Shopee is mostly 5-star reviews |
| Gold Test Set 1,500 and Fleiss' kappa | Not built | Out of scope this iteration; kappa cannot be computed |
| Neutral 4,982 after EDA | Not built | Augmentation out of scope this iteration |

Drops in this run: 4,583 empty after cleaning (star rating only), 590 MD5 exact duplicates (174 UIT-VSFC, 416 Shopee), 0 low-TTR. Labels: 10,832 positive, 7,711 negative, 773 neutral. 13 dropped duplicates had a different label from the kept row (the first row is kept).

## Defaults chosen where the brief was open (all in `vi_clean/config.py`)

- Stage order: NFC, regex, teencode (per text), then MD5 dedup, then TTR. The brief lists regex first and teencode last; NFC runs first so decomposed Vietnamese cannot defeat the regexes, and teencode runs before dedup so "ko" and "không" match.
- Lowercase on by default (training runs use `lowercase=False` because PhoBERT is case-sensitive); character runs capped at 2 (digits untouched); emoji kept; tone marks never stripped.
- MD5 key: casefolded text, punctuation removed, whitespace collapsed; first occurrence kept; exact match only.
- TTR: whitespace tokens; flagged when at least 5 tokens and unique/total below 0.3.
- Labels: UIT-VSFC 0/1/2 = negative/neutral/positive. Shopee 1-2 stars negative, 3 neutral, 4-5 positive (weak labels; `rating` column kept).
- Teencode: `vi_clean/data/teencode_v1.json`, hand-authored, whole-token, case-insensitive.

## Data provenance

UIT-VSFC came from the Hugging Face parquet conversion of `uitnlp/vietnamese_students_feedback` (the dataset card lists the license as unknown; check the original UIT NLP source before redistributing). It was converted once to `data/uit_vsfc/<split>/sents.txt` and `sentiments.txt` using a throwaway script; `pyarrow` is not a project dependency. `data/` is git-ignored.
