# Class-Weight Experiment, XLM-RoBERTa Comparison and Clean-up: Design

Status: draft for user review. Builds on PR #2 (`phobert-finetune`). Items A, B, C are the user's labels from the "what is left" discussion.

## Purpose
- **A.** Test whether softer class weights fix the Shopee Neutral failure (recall 0.17, precision 0.08; Neutral weight about 30x Positive).
- **B.** Fine-tune XLM-RoBERTa-base with the same protocol, so the report can compare it with PhoBERT (the research question).
- **C.** Close the known loose ends of PR #1 and #2: stale `docs/NOTES.md`, Excel-unfriendly CSVs, blank text in `/predict`.

## Non-goals
FastText baseline, EDA, ablation, t-SNE, web demo, Gold Set and kappa, new crawling, hyperparameter search beyond the weight power, test files (assumed to carry over; confirm). Anything already deferred in PR #2's minors that is not listed in C.

## Branch
New branch `phobert-experiments` cut from `phobert-finetune` (stacked PR #3). The `docs/NOTES.md` fix lands here too, not on PR #1's branch, because rebasing PR #2 onto a changed PR #1 is not worth it.

## C. Clean-up
- **NOTES.md:** replace the figures from the first run (16,865 rows, 1,000 Shopee reviews) with the real run (24,489 raw records, 19,316 clean = UIT-VSFC 16,001 + Shopee 3,315; Shopee 8,314 raw from 5 products; dropped 4,583 empty, 590 duplicates, 0 TTR; 13 label conflicts; Neutral 4.0%; difference -14,934).
- **Excel-safe files:** write `dataset.csv`, `segmented.csv` and the prediction CSVs as `utf-8-sig`, and read them with `utf-8-sig` (reads also accept files without a BOM). `product_id` becomes `s<shopid>_i<itemid>` (file stem if an id is missing) so Excel cannot turn it into a number. Existing baseline prediction files are rewritten once, old `a.b` to `sa_ib`.
- **Blank text in `/predict`:** when cleaning leaves an empty string, return `{"cleaned": "", "label": null, "probs": null}`; the rest of a batch is unaffected and order is kept.

## Run-aware training (infrastructure for A and B)
`TrainConfig` gains `run: str = "phobert-p1.0"`, `weight_power: float = 1.0`, `segment: bool = True`. Artifacts are per run:
- checkpoints: `checkpoints/<run>/{stage1,final}`; each saves `run.json` = `{"model_name", "segment", "weight_power"}`.
- predictions: `data/train/preds/<run>/`; report: `reports/<run>.json`.
- The existing baseline artifacts are moved into run `phobert-p1.0` (checkpoints and preds are git-ignored; `reports/metrics.json` is renamed with `git mv`).
- `class_weights(y, k, power=1.0)` returns `(N / (k * n_c)) ** power`; power 1.0 reproduces today's weights exactly.
- `train` CLI: `--run`, `--model`, `--weight-power`, `--no-segment`, `--shopee-dir`. `evaluate` CLI: `--run`.
- The API serves the run named by env `VI_RUN` (default `phobert-p1.0`); `Classifier.from_dir` reads `run.json` to decide whether to word-segment, so an XLM-R checkpoint is never fed PhoBERT-style segmented text.

## A. Weight-power experiment
- Candidates: power 1.0 (baseline, already trained), 0.5, 0.25. Each run is the full protocol (stage 1, 5 folds, final model).
- Selection: the candidate with the highest mean of the 5 per-fold validation Macro-F1 values in `train_log.json`, ties going to the larger power. Held-out (out-of-fold) scores are never used to choose.
- Reporting: all three candidates are reported side by side, including ones that do not win.

## B. XLM-RoBERTa
- Model `FacebookAI/xlm-roberta-base` (about 1.12 GB for `model.safetensors`, 5 MB `sentencepiece.bpe.model`, 9 MB `tokenizer.json`; download needs the user's approval at execution time).
- Same data, splits, seeds, epochs, learning rates and loss as PhoBERT, using the weight power selected in A. Input is the cleaned text with no word segmentation (SentencePiece), case kept.
- No tuning for XLM-R; if it underperforms, the report says it was run with PhoBERT's settings.

## Outputs
`reports/<run>.json` per run, plus `reports/comparison.md`: one table with UIT-VSFC test Macro-F1, Shopee out-of-fold Macro-F1 (interval), Neutral recall and precision, accuracy, and median latency for each run. The user's `main.tex` gets the real rows in place of its TBD markers for these results, with a backup first.

## Leak-free rules (unchanged)
Splits by whole product; weights and epoch selection from training and validation rows only; weight power chosen on validation only; held-out rows only predicted and scored.

## Known limits (reported, not hidden)
- Validation Neutral rows per fold are few, so the selection rule is noisy.
- One seed per run; differences of a point or two are within noise (the Shopee interval is about 0.42 to 0.62).
- XLM-R gets no tuning, so a loss is not evidence against it.
- Training time is unmeasured until the first new run finishes; the earlier "under 30 minutes" figure was a guess.

## Success criteria
Power 1.0 after migration reproduces the baseline report (Shopee Macro-F1 0.5363, UIT-VSFC final 0.7334); `/predict` returns `label: null` for blank text and works for a PhoBERT and an XLM-R checkpoint; `reports/comparison.md` lists every run actually trained; no unmeasured number appears in it.
