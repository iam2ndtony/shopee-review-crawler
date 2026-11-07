# PhoBERT Fine-tuning for Vietnamese Review Sentiment: Design

Status: draft for user review. Builds on the cleaning pipeline in `vi_clean/` (PR iam2ndtony/shopee-review-crawler#1).

## Purpose
A fine-tuned 3-class (negative / neutral / positive) sentiment model whose target domain is **Shopee / e-commerce reviews**, with honest per-source evaluation and a `POST /predict` endpoint so it is usable as a tool. Base model: `vinai/phobert-base` (the user's choice; replaces the brief's XLM-RoBERTa for this iteration).

## Non-goals
PhoBERT-v2, XLM-R, FastText baseline, EDA augmentation, t-SNE, web demo UI, Gold Test Set and Fleiss' kappa, hyperparameter search, near-duplicate detection. No test files (assumed to carry over from the cleaning iteration; confirm).

## Inputs
- `data/clean/dataset.csv` produced by `vi_clean.merge`, regenerated with `lowercase=False` (PhoBERT is case-sensitive). Current default-config run: 19,316 rows = UIT-VSFC 16,001 + Shopee 3,315; Neutral about 773 total, about 70 of them Shopee.
- Shopee raw reviews live outside this branch (main checkout `raw_reviews/`, 5 products). The training CLI takes `--shopee-dir`; raw data is not committed.

## Change to `vi_clean` (small)
`load_shopee` adds a `product_id` column (`"<shopid>.<itemid>"`, not personal data; NaN for UIT-VSFC). Schema becomes `text, label, source, rating, product_id`. `run_job` writes it through. Nothing else in the cleaning stages changes.

## Layout
New package `vi_train/`: `config.py` (`TrainConfig`, all hyperparameters named), `data.py` (load, segment, split), `train.py` (two-stage fine-tune), `evaluate.py` (metrics, latency), `predict.py` (checkpoint loader used by the API). `vi_clean/api.py` gains `/predict` and `/predict/batch`.

## Data preparation
- Word segmentation: `underthesea.word_tokenize(text, format="text")` on cleaned text, cached to `data/train/segmented.csv`. Known mismatch: PhoBERT was trained with VnCoreNLP segmentation; accepted for the dependency already being installed.
- Max length 128 tokens (Shopee median 16 words, 95th percentile 40).
- Splits, fixed seed 42, all made after the existing dedup:
  - UIT-VSFC: stratified by label, 80/10/10 train/val/test. Not comparable with published UIT-VSFC results (own split).
  - Shopee: **leave-one-product-out**, 5 folds by `product_id`. Within each fold's training products, 10% stratified by label is held out as validation for checkpoint selection.

## Training
- Stage 1: fine-tune PhoBERT-base on UIT-VSFC train (lr 2e-5, 3 epochs), select best epoch by UIT val Macro-F1. Run once and reused by all folds.
- Stage 2, per fold: initialize from stage 1, train on that fold's Shopee training products (lr 1e-5, 4 epochs), select best epoch by its Shopee val Macro-F1, evaluate on the held-out product.
- Final model: stage 2 on all 5 products (10% stratified val), saved to `checkpoints/final/`. Its quality is estimated by the leave-one-product-out scores, not its own data.
- Loss: cross-entropy with class weights `w_c = N / (K * n_c)` computed from the training portion of that stage only.
- AdamW, weight decay 0.01, 10% linear warmup, batch size 32, bf16 autocast. Plain PyTorch loop; no Trainer, no `accelerate`, no `scikit-learn`.
- Compute estimate (unverified): under 30 minutes in total on the RTX 5070 Ti.

## Evaluation (`reports/metrics.json`)
- Per source (UIT-VSFC test; Shopee pooled out-of-fold predictions): Macro-F1 (headline), per-class precision / recall / F1, confusion matrix, per-class counts, accuracy shown last.
- Any class with fewer than 30 evaluation rows is flagged `low_confidence`; Macro-F1 gets a 95% bootstrap interval (1,000 resamples, numpy).
- Latency: median and p95 milliseconds per review at batch sizes 1 and 32, after warm-up, reported for model forward and for end-to-end (segmentation + tokenize + forward).
- No Macro-F1 target is promised; results are reported as measured.

## Serving
- `POST /predict {text}` and `POST /predict/batch {texts}` (1-64 items) return `{cleaned, label, probs}`; input goes through the same `clean_texts` with `lowercase=False`, then segmentation and the model. `/clean` keeps its default config.
- The checkpoint loads lazily once, behind a lock; GPU when available. Returns 503 until `checkpoints/final/` exists.

## Dependencies
Add `torch`, `transformers`, `underthesea` to `requirements.txt`. Downloading `vinai/phobert-base` weights (about 540 MB) from Hugging Face needs the user's approval at execution time.

## Leak-free rules
Dedup and cleaning precede splitting; Shopee splits are by whole product; class weights and checkpoint selection use training and validation parts only; the test part is read only by `evaluate.py`; the teencode dictionary is hand-authored.

## Known limits (reported, not hidden)
- Shopee Neutral is about 70 rows, so its scores carry wide intervals even pooled.
- Shopee labels are weak (stars); star-versus-text mismatches are not corrected.
- 5 products from few categories; stars-only reviews (about 55% of crawled data) are unusable.
- The held-out sets are not a human-labeled Gold Set.
- Residual leakage risk: near-duplicate UIT-VSFC sentences across train and test are not detected (exact duplicates only).
- Segmentation differs slightly from PhoBERT's pretraining segmenter.

## Success criteria
`python -m vi_train.train` and `python -m vi_train.evaluate` run end to end on this machine; `reports/metrics.json` contains every field above; `/predict` returns a label and probabilities for sample reviews; no test row is used in training, validation or class weights.
