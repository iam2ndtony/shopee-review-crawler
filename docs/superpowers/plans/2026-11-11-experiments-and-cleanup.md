# Class-Weight Experiment, XLM-RoBERTa Comparison and Clean-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the PR #1/#2 loose ends (C), test softer class weights for Shopee Neutral (A), and fine-tune XLM-RoBERTa-base under the same protocol (B), ending in one honest comparison table and a filled-in report.

**Architecture:** Training artifacts become per-run (`checkpoints/<run>/`, `data/train/preds/<run>/`, `reports/<run>.json`) with a `run.json` beside each checkpoint saying how to feed it text. `class_weights` gains a power. The existing `vi_train.train` / `vi_train.evaluate` / API serve any run. A small `vi_train.compare` builds the comparison table from the report files.

**Tech Stack:** Python 3.13, PyTorch (installed), transformers (installed), underthesea (installed), pandas, numpy, fastapi. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-11-experiments-and-cleanup-design.md`

## Global Constraints

- Work on new branch `phobert-experiments` cut from `phobert-finetune`; commit the spec and this plan with Task 1. Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- No test files (assumed from earlier iterations). Each task is verified by running the given command and comparing output. Run Python with `PYTHONUTF8=1`.
- No new dependencies. Do not commit `data/`, `checkpoints/` or raw reviews; `reports/*.json` and `reports/comparison.md` are committed.
- Downloading `FacebookAI/xlm-roberta-base` (about 1.12 GB safetensors plus 14 MB tokenizer files) needs the user's approval: stop and ask in Task 6 before the first download.
- The weight power is chosen only from per-fold validation Macro-F1 in `train_log.json`, never from out-of-fold or UIT test scores. Every trained candidate is reported.
- Every number in `reports/comparison.md` and in the user's `main.tex` comes from a `reports/*.json` file. Never invent or round-up a figure; unmeasured items stay as red `[TBD]`.
- Shopee review folder for all runs: `C:\Users\admin\CODE\shopee_review_crawler\raw_reviews` (main checkout, 5 products).
- `main.tex` (`C:\Users\admin\Downloads\main.tex`) is outside the repo: back it up before editing, and only touch the XLM-R, weight and result rows.
- Pushing the branch and opening PR #3 is a stop point: ask the user first.

## Review Focus

1. The weight power is picked from validation scores only, and a candidate that does not win is still in the table.
2. After the artifact migration, `--weight-power 1.0` reproduces the baseline exactly: Shopee out-of-fold Macro-F1 0.5363 and UIT-VSFC final 0.7334.
3. A CSV with a BOM and one without both read with header `text` (never `\ufefftext`), and `product_id` stays a string like `s1103428920_i46913118781`.
4. `/predict` with blank or whitespace text returns `label: null` (single and batch, order kept) and does not load a checkpoint when every text is blank.
5. An XLM-R checkpoint is served without word segmentation (via `run.json`), and a checkpoint directory with no `run.json` is still served as PhoBERT.

---

### Task 1: Branch, docs and NOTES.md

**Files:**
- Modify: `docs/NOTES.md`
- Create (commit with this task): the spec and this plan

**Interfaces:**
- Produces: branch `phobert-experiments`; `docs/NOTES.md` states the real run.

- [ ] **Step 1:** `git switch -c phobert-experiments`.
- [ ] **Step 2: Update `docs/NOTES.md`** from `data/train/clean/report.json` (read it; do not retype from memory): rows in (UIT-VSFC 16,175; Shopee 8,314 from 5 products), dropped (empty, MD5 duplicates, TTR), label conflicts, final per source and label, Neutral share, difference from 34,250. Keep the "defaults chosen" section; add one line that `lowercase=False` is used for training runs.
- [ ] **Step 3: Verify**

Run: `grep -c "16,865" docs/NOTES.md; grep -c "19,316" docs/NOTES.md`
Expected: `0` then at least `1`.

- [ ] **Step 4: Commit** `git add docs && git commit -m "docs: update NOTES with the 5-product run; add experiments spec and plan"`

### Task 2: Excel-safe CSVs and text-safe product_id

**Files:**
- Modify: `vi_clean/merge.py`, `vi_train/data.py`, `vi_train/train.py`, `vi_train/evaluate.py`

**Interfaces:**
- Produces: `product_id` is `"s<shopid>_i<itemid>"` (file stem if an id is missing). `dataset.csv`, `segmented.csv` and all prediction CSVs are written with `encoding="utf-8-sig"` and read with `encoding="utf-8-sig"` (a file without a BOM still reads).

- [ ] **Step 1:** In `load_shopee` build `f"s{shopid}_i{itemid}"`; write `dataset.csv` with `utf-8-sig` (keep the atomic `os.replace`).
- [ ] **Step 2:** Switch every `read_csv` / `to_csv` of those files in `data.py`, `train.py` (`_preds` callers) and `evaluate.py` (`load`) to `utf-8-sig`.
- [ ] **Step 3: Rewrite the baseline predictions once**: in `data/train/preds/shopee_oof.csv` change `product_id` values matching `^(\d+)\.(\d+)$` to `s\1_i\2` and save as `utf-8-sig`; re-save the other three CSVs as `utf-8-sig` (content unchanged).
- [ ] **Step 4: Verify the data path**

Run: `PYTHONUTF8=1 python -c "import dataclasses; from pathlib import Path; from vi_train.config import TrainConfig; from vi_train.data import prepare; c=dataclasses.replace(TrainConfig(), shopee_dir=Path(r'C:\Users\admin\CODE\shopee_review_crawler\raw_reviews')); df=prepare(c); print(open(c.work_dir/'clean'/'dataset.csv','rb').read(3)==b'\xef\xbb\xbf', list(df.columns)[:2], df.product_id.dropna().iloc[0], df.product_id.dropna().str.fullmatch(r's\d+_i\d+').all(), len(df))"`
Expected: `True`, `['text', 'label']`, an id starting with `s`, `True`, `19316`.

- [ ] **Step 5: Verify the evaluation is unchanged**

Run: `PYTHONUTF8=1 python -m vi_train.evaluate > /dev/null && python -c "import json; r=json.load(open('reports/metrics.json',encoding='utf-8')); print(r['shopee_oof']['macro_f1'], r['uit_vsfc_test_final']['macro_f1'], sorted(r['shopee_per_product'])[0])"`
Expected: `0.5363 0.7334` and a key starting with `s`.

- [ ] **Step 6: Commit** `git add vi_clean vi_train reports/metrics.json && git commit -m "fix: write CSVs as utf-8-sig and use a text-safe product_id"`

### Task 3: Blank text in `/predict`

**Files:**
- Modify: `vi_clean/api.py`

**Interfaces:**
- Produces: when cleaning leaves `""`, that item returns `{"cleaned": "", "label": None, "probs": None}`; other items in the batch are classified as before and results keep input order. A request whose texts are all blank never loads the checkpoint (so no 503).

- [ ] **Step 1: Implement** in `_predict`: classify only the non-empty cleaned texts, then merge back by index.
- [ ] **Step 2: Verify**

Run: `PYTHONUTF8=1 python -c "from fastapi.testclient import TestClient; import vi_clean.api as a; c=TestClient(a.app); print(c.post('/predict',json={'text':'   '}).json()); r=c.post('/predict/batch',json={'texts':['Hàng đẹp','  ','Dùng tệ quá']}).json()['results']; print([x['label'] for x in r])"`
Expected: `{'cleaned': '', 'label': None, 'probs': None}` then `['positive', None, 'negative']`.

- [ ] **Step 3: Verify no checkpoint is needed for all-blank input**

Run: `mv checkpoints/final checkpoints/final_hold; PYTHONUTF8=1 python -c "from fastapi.testclient import TestClient; import vi_clean.api as a; c=TestClient(a.app); print(c.post('/predict',json={'text':' '}).status_code, c.post('/predict',json={'text':'tốt'}).status_code)"; mv checkpoints/final_hold checkpoints/final`
Expected: `200 503`, and `checkpoints/final` restored afterwards.

- [ ] **Step 4: Commit** `git add vi_clean/api.py && git commit -m "fix: /predict returns a null label for blank text"`

### Task 4: Run-aware training and weight power

**Files:**
- Modify: `vi_train/config.py`, `vi_train/data.py`, `vi_train/train.py`, `vi_train/predict.py`, `vi_train/evaluate.py`, `vi_clean/api.py`, `README.md`
- Move (git-ignored data, tracked report): baseline artifacts into run `phobert-p1.0`

**Interfaces:**
- Produces in `config.py`: `TrainConfig` fields `run: str = "phobert-p1.0"`, `weight_power: float = 1.0`, `segment: bool = True`, `reports_dir: Path = ROOT/"reports"` (replaces `report_path`); properties `run_ckpt -> Path` (`ckpt_dir/run`), `run_preds -> Path` (`work_dir/"preds"/run`), `run_report -> Path` (`reports_dir/f"{run}.json"`).
- Produces in `data.py`: `class_weights(y: np.ndarray, k: int, power: float = 1.0) -> np.ndarray` returning `(N / (k * n_c)) ** power`; `prepare(cfg)` sets column `seg` to the segmented text when `cfg.segment` else to `text` (so `seg` always means "model input").
- Produces in `train.py`: `main(argv)` accepts `--run`, `--model`, `--weight-power`, `--no-segment`, `--shopee-dir`; checkpoints and predictions go under `cfg.run_ckpt` / `cfg.run_preds`; `_save` also writes `run.json` = `{"model_name", "segment", "weight_power"}` in each saved dir.
- Produces in `predict.py`: `Classifier.segment: bool`, read from `run.json` in the checkpoint dir (default `True` when the file is absent).
- Produces: `evaluate.main` accepts `--run` and reads/writes that run's paths; `_latency` skips word segmentation when `clf.segment` is false. The API serves the run in env `VI_RUN` (default `TrainConfig().run`) and segments only when `clf.segment`.

- [ ] **Step 1:** Implement the config, `class_weights`, `prepare` and CLI changes above.
- [ ] **Step 2: Migrate the baseline artifacts**: move `checkpoints/stage1` and `checkpoints/final` into `checkpoints/phobert-p1.0/`, move `data/train/preds/*` into `data/train/preds/phobert-p1.0/`, `git mv reports/metrics.json reports/phobert-p1.0.json`, and write `run.json` (`vinai/phobert-base`, `true`, `1.0`) into both checkpoint dirs.
- [ ] **Step 3: Verify the weights**

Run: `python -c "import numpy as np; from vi_train.data import class_weights; y=np.array([0]*5+[1]*2+[2]*3); print(class_weights(y,3).round(3), class_weights(y,3,0.5).round(3))"`
Expected: `[0.667 1.667 1.111] [0.816 1.291 1.054]`.

- [ ] **Step 4: Verify the baseline reproduces**

Run: `PYTHONUTF8=1 python -m vi_train.evaluate --run phobert-p1.0 > /dev/null && python -c "import json; r=json.load(open('reports/phobert-p1.0.json',encoding='utf-8')); print(r['shopee_oof']['macro_f1'], r['uit_vsfc_test_final']['macro_f1'])"`
Expected: `0.5363 0.7334`.

- [ ] **Step 5: Verify serving and flags**

Run: `PYTHONUTF8=1 python -c "import sys; from fastapi.testclient import TestClient; import vi_clean.api as a; c=TestClient(a.app); print('torch' in sys.modules); print(c.post('/predict',json={'text':'Hàng đẹp, giao nhanh'}).json()['label'])"; python -m vi_train.train --help`
Expected: `False`, `positive`, and the help lists `--run`, `--model`, `--weight-power`, `--no-segment`, `--shopee-dir`.

- [ ] **Step 6:** Add the new flags and `VI_RUN` to `README.md`.
- [ ] **Step 7: Commit** `git add vi_train vi_clean README.md reports && git commit -m "feat: run-aware training with a class-weight power"`

### Task 5: Weight-power experiment (A)

**Files:**
- Create: `reports/phobert-p0.5.json`, `reports/phobert-p0.25.json` (generated)

**Interfaces:**
- Consumes: `python -m vi_train.train` / `evaluate` from Task 4.
- Produces: runs `phobert-p0.5` and `phobert-p0.25`, each with the full protocol, plus the selected power recorded in the commit message and in Task 7's table.

- [ ] **Step 1: Train** (long; run in the background and note the wall-clock time of each run, the first real timing measurement).

Run: `PYTHONUTF8=1 python -u -m vi_train.train --shopee-dir C:\Users\admin\CODE\shopee_review_crawler\raw_reviews --run phobert-p0.5 --weight-power 0.5` then the same with `--run phobert-p0.25 --weight-power 0.25`
Expected: each finishes with five folds and a `done {...}` line; creates `checkpoints/<run>/{stage1,final}` and `data/train/preds/<run>/`.

- [ ] **Step 2: Evaluate both**

Run: `PYTHONUTF8=1 python -m vi_train.evaluate --run phobert-p0.5` then `--run phobert-p0.25`
Expected: `reports/phobert-p0.5.json` and `reports/phobert-p0.25.json` exist with every key the baseline report has.

- [ ] **Step 3: Select the power by validation only**

Run: `python -c "import json,statistics as s; [print(r, round(s.mean(json.load(open(f'data/train/preds/{r}/train_log.json'))['folds'].values()),4)) for r in ('phobert-p1.0','phobert-p0.5','phobert-p0.25')]"`
Expected: three lines with a mean validation Macro-F1 each. The highest wins; a tie goes to the larger power. Record the winner and all three values; do not look at out-of-fold scores to decide.

- [ ] **Step 4: Commit** `git add reports && git commit -m "feat: weight-power experiment runs (selected power: <p>)"`

### Task 6: XLM-RoBERTa (B)

**Files:**
- Create: `reports/xlmr-p<selected>.json` (generated)

**Interfaces:**
- Consumes: Task 4's `--model` / `--no-segment`, the power selected in Task 5.
- Produces: run `xlmr-p<selected>` with the same protocol; its checkpoint's `run.json` has `segment: false`.

- [ ] **Step 1: Stop and ask the user** to approve downloading `FacebookAI/xlm-roberta-base`: `model.safetensors` 1,115,567,652 bytes, `sentencepiece.bpe.model` 5,069,051, `tokenizer.json` 9,096,718, plus config files, about 1.13 GB into the Hugging Face cache. Do not fetch `pytorch_model.bin`, `model.onnx`, `flax_model.msgpack` or `tf_model.h5`.
- [ ] **Step 2: Smoke-load** the tokenizer and model and run one forward pass of a Vietnamese sentence with `truncation=True, max_length=128`.

Run: `PYTHONUTF8=1 python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; t=AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base'); m=M.from_pretrained('FacebookAI/xlm-roberta-base', num_labels=3); e=t(['Hàng đẹp, giao nhanh'], truncation=True, max_length=128, return_tensors='pt'); print(type(m).__name__, m(**e).logits.shape)"`
Expected: `XLMRobertaForSequenceClassification torch.Size([1, 3])`.

- [ ] **Step 3: Train and evaluate**

Run: `PYTHONUTF8=1 python -u -m vi_train.train --shopee-dir C:\Users\admin\CODE\shopee_review_crawler\raw_reviews --run xlmr-p<selected> --model FacebookAI/xlm-roberta-base --no-segment --weight-power <selected>` then `PYTHONUTF8=1 python -m vi_train.evaluate --run xlmr-p<selected>`
Expected: five folds finish; `reports/xlmr-p<selected>.json` exists; `checkpoints/xlmr-p<selected>/final/run.json` has `"segment": false`. If out-of-memory, lower `batch_size` for this run only and ledger a ruling; if the loss does not decrease in epoch 1, report it as measured rather than tuning.

- [ ] **Step 4: Verify serving without segmentation**

Run: `VI_RUN=xlmr-p<selected> PYTHONUTF8=1 python -c "from fastapi.testclient import TestClient; import vi_clean.api as a; c=TestClient(a.app); print(c.post('/predict',json={'text':'Hàng đẹp, giao nhanh'}).json())"`
Expected: a label and three probabilities summing to about 1, with no error.

- [ ] **Step 5: Commit** `git add reports && git commit -m "feat: XLM-RoBERTa run with the PhoBERT protocol"`

### Task 7: Comparison, report and PR

**Files:**
- Create: `vi_train/compare.py`, `reports/comparison.md` (generated)
- Modify: `README.md`; user's `C:\Users\admin\Downloads\main.tex` (after a backup copy `main.backup2.tex`)

**Interfaces:**
- Consumes: every `reports/*.json`.
- Produces: `main() -> None` writes `reports/comparison.md`: one row per run with weight power, model, UIT-VSFC test Macro-F1 (final model), Shopee out-of-fold Macro-F1 with its interval, Neutral precision and recall, accuracy (last column), and batch-1 and batch-32 median latency; followed by the report's `notes` lines once.

- [ ] **Step 1: Implement `compare.py`** reading only the report files.
- [ ] **Step 2: Verify**

Run: `PYTHONUTF8=1 python -m vi_train.compare && python -c "print(open('reports/comparison.md',encoding='utf-8').read())"`
Expected: a table with one row per run actually trained (`phobert-p1.0`, `phobert-p0.5`, `phobert-p0.25`, `xlmr-p<selected>`); spot-check two cells against the JSON files.

- [ ] **Step 3: Update `main.tex`**: back it up, then in the "Implementation Status and Preliminary Results" section add the XLM-R and weight-power rows from `comparison.md`, replace the "XLM-RoBERTa has not been trained" and related text with measured results, and keep `[TBD]` for FastText, ablation, t-SNE and VRAM. Say plainly where XLM-R lost or won and that XLM-R was not tuned.
- [ ] **Step 4: Verify the report text**

Run: `grep -c "TBD" "C:\Users\admin\Downloads\main.tex"` and compare each new number in the XLM-R and weight rows against `reports/*.json`.
Expected: fewer `[TBD]` markers than the previous 12, none removed for unmeasured items.

- [ ] **Step 5: Commit** `git add vi_train/compare.py reports README.md && git commit -m "feat: comparison table across runs"`
- [ ] **Step 6: Stop and ask** the user before pushing `phobert-experiments` and opening PR #3 with base `phobert-finetune`.
