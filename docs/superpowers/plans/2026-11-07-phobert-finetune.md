# PhoBERT Fine-tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A two-stage fine-tuned PhoBERT-base sentiment model for Shopee reviews, with leave-one-product-out evaluation and a `POST /predict` endpoint.

**Architecture:** `vi_train/` reads the cleaned dataset (regenerated with lowercase off, plus a new `product_id` column), word-segments it, splits it, fine-tunes PhoBERT on UIT-VSFC then on Shopee, writes prediction files, and `evaluate.py` turns those into `reports/metrics.json`. `vi_clean/api.py` loads the final checkpoint lazily for `/predict`.

**Tech Stack:** Python 3.13, PyTorch (CUDA, installed), transformers (installed), underthesea (installed), pandas, numpy, fastapi.

**Spec:** `docs/superpowers/specs/2026-11-07-phobert-finetune-design.md`

## Global Constraints

- Work on a new branch `phobert-finetune` cut from `worktree-vi-clean-pipeline` (PR #1 is not merged); commit the spec and this plan with Task 1.
- No test files (assumed from the cleaning iteration). Each task is verified by running the given command and comparing output.
- Dependencies: only `torch`, `transformers`, `underthesea` added to `requirements.txt`. No `scikit-learn`, no `accelerate`, no Trainer; plain PyTorch loop; metrics in numpy.
- Downloading `vinai/phobert-base` (about 540 MB, Hugging Face) needs the user's approval: stop and ask in Task 4 before the first download.
- Do not change the cleaning stages in `vi_clean/stages.py`. Do not commit `data/`, `checkpoints/` or raw reviews; `reports/metrics.json` may be committed.
- Shopee `author_username` / `userid` must never be loaded; `product_id` is `"<shopid>.<itemid>"` only.
- Fixed seed 42 everywhere; all hyperparameters are `TrainConfig` fields with the spec's values.
- Class weights, checkpoint selection and CV folds may use only training / validation rows; held-out rows are read only to predict and to score.

## Review Focus

1. A product id never appears in both training and held-out rows of a Shopee fold.
2. A class with 0-2 rows in a split (Shopee Neutral) must not crash or give NaN weights: zero rows raises a clear `ValueError` naming the class; tiny classes keep at least 1 row in the training part.
3. Held-out rows are disjoint from training and validation rows, asserted in code, never silently overlapping.
4. `/predict` with empty or whitespace text, or text longer than `max_len`, returns a label (truncated), not a 500.
5. `/predict` before `checkpoints/final/` exists returns 503, not 500, and `/clean` still works without torch imported.

---

### Task 1: `product_id` in the cleaning job

**Files:**
- Modify: `vi_clean/merge.py` (`COLUMNS`, `load_uit_vsfc`, `load_shopee`)
- Create (commit with this task): the two docs for this iteration

**Interfaces:**
- Produces: `dataset.csv` and loader frames have columns `text, label, source, rating, product_id`; `product_id` is `"<shopid>.<itemid>"` for Shopee (file stem if either id is missing) and NaN for UIT-VSFC.

- [ ] **Step 1: Create the branch** `git switch -c phobert-finetune`.
- [ ] **Step 2: Add `product_id`** to `COLUMNS`, set `np.nan` in `load_uit_vsfc`, and build it from `item.get("shopid")` / `item.get("itemid")` in `load_shopee`.
- [ ] **Step 3: Verify**

Run: `PYTHONUTF8=1 python -c "from pathlib import Path; import dataclasses; from vi_clean.config import CleanConfig; from vi_clean.merge import load_shopee, load_uit_vsfc; c=dataclasses.replace(CleanConfig(), shopee_dir=Path(r'C:\Users\admin\CODE\shopee_review_crawler\raw_reviews')); s=load_shopee(c); print(list(s.columns), s.product_id.nunique(), len(s), load_uit_vsfc(c).product_id.isna().all())"`
Expected: columns end with `'product_id'`, `5`, `8314`, `True`.

- [ ] **Step 4: Commit** `git add vi_clean/merge.py docs && git commit -m "feat: add product_id to merged dataset"`

### Task 2: Training config, data preparation and splits

**Files:**
- Create: `vi_train/__init__.py` (empty), `vi_train/config.py`, `vi_train/data.py`

**Interfaces:**
- Consumes: `run_job`, `CleanConfig` (Task 1 schema).
- Produces in `config.py`: frozen dataclass `TrainConfig` with `model_name="vinai/phobert-base"`, `labels=("negative","neutral","positive")`, `max_len=128`, `batch_size=32`, `stage1_lr=2e-5`, `stage1_epochs=3`, `stage2_lr=1e-5`, `stage2_epochs=4`, `weight_decay=0.01`, `warmup_frac=0.1`, `val_frac=0.1`, `seed=42`, `low_conf_min=30`, `bootstrap_n=1000`, `shopee_dir: Path = ROOT/"raw_reviews"`, `work_dir: Path = ROOT/"data"/"train"`, `ckpt_dir: Path = ROOT/"checkpoints"`, `report_path: Path = ROOT/"reports"/"metrics.json"`.
- Produces in `data.py`:
  - `segment(text: str) -> str` (underthesea `word_tokenize(..., format="text")`; empty in, empty out)
  - `prepare(cfg: TrainConfig) -> pd.DataFrame` (columns `text,label,source,rating,product_id,seg,y`; runs `run_job` with `lowercase=False`, `shopee_dir=cfg.shopee_dir`, `out_dir=cfg.work_dir/"clean"`; caches segmentation in `cfg.work_dir/"segmented.csv"`)
  - `stratified_split(df: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]` returns `(rest, held)`; per class `held` gets `round(frac*n)` rows, at least 1 when n >= 2, 0 when n == 1
  - `split_uit(df, cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]` (train, val, test; 80/10/10 stratified on `source == "uit_vsfc"`)
  - `shopee_folds(df, cfg) -> list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]]` (held-out product id, train, val, test; val = `stratified_split(train_products, cfg.val_frac)`)
  - `shopee_final(df, cfg) -> tuple[pd.DataFrame, pd.DataFrame]` (train, val over all Shopee rows)
  - `class_weights(y: np.ndarray, k: int) -> np.ndarray` returning `N / (k * n_c)`; raises `ValueError` naming the class index when `n_c == 0`

- [ ] **Step 1: Write `config.py`**.
- [ ] **Step 2: Implement the functions in `data.py`** as above; sorting by index before splitting and `np.random.default_rng(seed)` for shuffles.
- [ ] **Step 3: Verify splits and leakage**

Run: `PYTHONUTF8=1 python -c "import dataclasses; from pathlib import Path; from vi_train.config import TrainConfig; from vi_train.data import *; c=dataclasses.replace(TrainConfig(), shopee_dir=Path(r'C:\Users\admin\CODE\shopee_review_crawler\raw_reviews')); df=prepare(c); tr,va,te=split_uit(df,c); print(len(tr),len(va),len(te)); f=shopee_folds(df,c); print(len(f)); print(all(set(a.product_id).isdisjoint(set(d.product_id)) and set(b.index).isdisjoint(set(a.index)) and set(b.index).isdisjoint(set(d.index)) for _,a,b,d in f)); print(class_weights(tr.y.to_numpy(),3).round(2))"`
Expected: UIT sizes about `12800 1600 1600` (80/10/10 of 16,001), `5`, `True`, three weights with Neutral the largest.

- [ ] **Step 4: Commit** `git add vi_train && git commit -m "feat: add training config, segmentation and splits"`

### Task 3: Metrics and prediction helpers

**Files:**
- Create: `vi_train/evaluate.py` (metrics part only for now), `vi_train/predict.py`

**Interfaces:**
- Consumes: `TrainConfig`.
- Produces in `evaluate.py`:
  - `macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float` (classes absent from both y_true and y_pred count as F1 0)
  - `report(y_true: np.ndarray, y_pred: np.ndarray, cfg: TrainConfig) -> dict` with keys `macro_f1`, `macro_f1_ci` (95% bootstrap, `cfg.bootstrap_n` resamples, seeded), `per_class` (label -> `precision, recall, f1, n`), `confusion` (list of lists), `accuracy`, `low_confidence` (labels with `n < cfg.low_conf_min`)
- Produces in `predict.py`:
  - `predict_probs(model, tokenizer, texts: list[str], max_len: int, batch_size: int, device: str) -> np.ndarray` softmax probabilities, shape `(len(texts), 3)`, truncation on, empty list returns shape `(0, 3)`
  - `class Classifier` with `from_dir(path: Path, cfg: TrainConfig) -> Classifier` and `probs(seg_texts: list[str]) -> np.ndarray`; picks CUDA when available, eval mode, no grad

- [ ] **Step 1: Implement `macro_f1` and `report`** in numpy.
- [ ] **Step 2: Verify metrics** on a hand case.

Run: `python -c "import numpy as np; from vi_train.config import TrainConfig; from vi_train.evaluate import macro_f1, report; y=np.array([0,0,1,1,2,2]); p=np.array([0,1,1,1,2,0]); print(round(macro_f1(y,p,3),3)); r=report(y,p,TrainConfig()); print(r['per_class']['neutral'], r['low_confidence'], r['accuracy'])"`
Expected: `0.656`; neutral `precision 0.667, recall 1.0, f1 0.8, n 2`; all three labels in `low_confidence`; accuracy `0.667`.

- [ ] **Step 3: Implement `predict_probs` and `Classifier`**; no verification run until Task 4 has a checkpoint.
- [ ] **Step 4: Commit** `git add vi_train && git commit -m "feat: add metrics and prediction helpers"`

### Task 4: Two-stage training with leave-one-product-out

**Files:**
- Create: `vi_train/train.py`

**Interfaces:**
- Consumes: Task 2 data functions, Task 3 `predict_probs` and `macro_f1`.
- Produces:
  - `train_stage(model, tokenizer, train_df: pd.DataFrame, val_df: pd.DataFrame, lr: float, epochs: int, cfg: TrainConfig, device: str) -> float` trains with class-weighted cross-entropy (weights from `train_df` only), AdamW with `cfg.weight_decay`, linear warmup `cfg.warmup_frac`, bf16 autocast on CUDA, keeps the best epoch's weights by val Macro-F1, restores them into `model`, returns that val Macro-F1
  - `main(argv: list[str] | None = None) -> None` with `--shopee-dir PATH`; runs stage 1 (UIT train/val), saves `cfg.ckpt_dir/"stage1"`, writes UIT-test predictions of stage 1, then per Shopee fold loads stage-1 weights, runs stage 2 and predicts the held-out product; then trains the final model on `shopee_final`, saves `cfg.ckpt_dir/"final"`, writes UIT-test predictions of the final model; asserts held-out indices are disjoint from train and val
  - Output files under `cfg.work_dir/"preds"`: `uit_test_stage1.csv`, `uit_test_final.csv`, `shopee_oof.csv` (columns `text, seg, y, pred, product_id`) and `train_log.json` (best val Macro-F1 per stage and fold)

- [ ] **Step 1: Stop and ask the user** to approve downloading `vinai/phobert-base` (about 540 MB, Hugging Face) before any download. If `AutoTokenizer` cannot load PhoBERT under the installed transformers version, ledger a ruling and fall back to `PhobertTokenizer` explicitly.
- [ ] **Step 2: Implement `train_stage` and `main`** per the interface; seed `torch` and numpy at the start of each stage.
- [ ] **Step 3: Smoke run on a tiny subset**, then the full run.

Run: `PYTHONUTF8=1 python -m vi_train.train --shopee-dir C:\Users\admin\CODE\shopee_review_crawler\raw_reviews`
Expected: prints one line per epoch with val Macro-F1; finishes with all five folds; creates `checkpoints/stage1`, `checkpoints/final`, the three prediction CSVs and `train_log.json`; no assertion errors.

- [ ] **Step 4: Commit** `git add vi_train/train.py && git commit -m "feat: add two-stage PhoBERT training with leave-one-product-out"`

### Task 5: Evaluation report and latency

**Files:**
- Modify: `vi_train/evaluate.py` (add `main`)

**Interfaces:**
- Consumes: `report`, prediction CSVs and `train_log.json` from Task 4, `Classifier` and `segment`.
- Produces: `main() -> None` writes `cfg.report_path` with keys `uit_vsfc_test_stage1`, `uit_vsfc_test_final`, `shopee_oof` (each a `report(...)` dict plus `n`), `shopee_per_product` (Macro-F1 per held-out product), `training` (from `train_log.json`), `latency` (`model_forward` and `end_to_end`, each with `batch1` and `batch32` giving `median_ms` and `p95_ms` per review, measured on 200 `shopee_oof.csv` rows after 10 warm-up calls, with CUDA synchronize), and `notes` (always lists: not a Gold Set; Shopee Neutral count; weak star labels).

- [ ] **Step 1: Implement `main`** per the interface; accuracy appears last in each dict.
- [ ] **Step 2: Run**

Run: `PYTHONUTF8=1 python -m vi_train.evaluate`
Expected: `reports/metrics.json` exists with every key above; `shopee_oof.n` equals the number of Shopee rows in the dataset; `low_confidence` lists Neutral for Shopee if its count is under 30.

- [ ] **Step 3: Commit** `git add vi_train/evaluate.py reports/metrics.json && git commit -m "feat: add evaluation report and latency measurement"`

### Task 6: `/predict` endpoints and docs

**Files:**
- Modify: `vi_clean/api.py`, `requirements.txt` (add `torch`, `transformers`, `underthesea`), `README.md`

**Interfaces:**
- Consumes: `Classifier`, `segment`, `clean_texts`, `CleanConfig`.
- Produces: `POST /predict {text: str}` -> `{cleaned: str, label: str, probs: {negative, neutral, positive}}`; `POST /predict/batch {texts: list[str]}` (1-64 items) -> `{results: [...]}`; cleaning uses `dataclasses.replace(cfg, lowercase=False)`; torch and the classifier are imported and loaded lazily on first call behind a `threading.Lock`; a missing `checkpoints/final/` returns HTTP 503 with a clear message. The `/clean` routes and startup do not import torch.

- [ ] **Step 1: Implement the routes** with lazy loading.
- [ ] **Step 2: Verify**

Run: `PYTHONUTF8=1 python -c "import sys; from fastapi.testclient import TestClient; import vi_clean.api as a; c=TestClient(a.app); print('torch' in sys.modules); r=c.post('/predict',json={'text':'Sản phẩm dùng rất tốt, giao hàng nhanh'}); print(r.status_code, r.json()); print(c.post('/predict',json={'text':'   '}).status_code); print(c.post('/predict/batch',json={'texts':[]}).status_code)"`
Expected: `False` (torch not imported at startup), `200` with a `label` and three probabilities summing to about 1, `200` for the whitespace text, `422` for the empty batch. With `checkpoints/final` temporarily renamed, `/predict` returns `503`.

- [ ] **Step 3: Update `README.md`** with how to run `vi_train.train`, `vi_train.evaluate` and `/predict`.
- [ ] **Step 4: Commit** `git add vi_clean/api.py requirements.txt README.md && git commit -m "feat: add /predict endpoints"`
