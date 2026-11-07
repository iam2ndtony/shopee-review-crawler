# Vietnamese Review Cleaning Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `vi_clean` package that merges UIT-VSFC and Shopee reviews, runs the 5-stage cleaning pipeline, and exposes it through a minimal FastAPI app.

**Architecture:** Five stage functions in `stages.py` driven by a frozen `CleanConfig`; a teencode dictionary loaded from a versioned JSON file; `merge.py` loads both sources into one schema, applies the stages, writes a CSV and a report; `api.py` wraps the same functions.

**Tech Stack:** Python 3.13, pandas, numpy, fastapi, uvicorn.

**Spec:** `docs/superpowers/specs/2026-11-07-vi-clean-pipeline-design.md`

## Global Constraints

- No tests (user instruction). Each task is verified by running the given command and comparing output; never add test files.
- Dependencies: pandas, numpy, fastapi, uvicorn only. No pyarrow, no tokenizer, no Crawl4AI.
- Never strip Vietnamese tone marks; all file I/O uses `encoding="utf-8"`.
- Do not modify `shopee_crawler_persistent.py` or `clean_reviews.py`.
- No Gold Set, no split, no statistic or dictionary derived from the data. Do not tune any threshold to reach 34,250.
- Every open parameter is a named `CleanConfig` field with the default stated in the spec.
- Do not commit `data/`, and do not load Shopee `author_username` or `userid`.

## Review Focus

1. Teencode must match whole tokens only: `"ko"` becomes `"không"` but `"kobe"` is unchanged.
2. Same text with different labels across sources: keep first, count the conflicts in the report.
3. `None`, empty or whitespace-only text (API input and Shopee `comment: null`) yields an empty cleaned text and is dropped by the job, never a crash.
4. Decomposed input (`"é"`) is composed to NFC before the regexes run.
5. Missing `data/uit_vsfc/` or `raw_reviews/*.json` gives a clear error naming the path, not a traceback from pandas.

---

### Task 1: Config and teencode dictionary

**Files:**
- Create: `vi_clean/__init__.py` (empty), `vi_clean/config.py`, `vi_clean/teencode.py`, `vi_clean/data/teencode_v1.json`

**Interfaces:**
- Produces: `CleanConfig` (frozen dataclass) with fields `stage_order: tuple[str, ...] = ("nfc", "regex", "teencode")`, `lowercase: bool = True`, `max_char_repeat: int = 2`, `min_tokens_for_ttr: int = 5`, `ttr_min: float = 0.3`, `expected_total: int = 34250`, `star_to_label: dict[int, str]` (1,2 negative; 3 neutral; 4,5 positive), `uit_label_map: dict[int, str]` (0 negative, 1 neutral, 2 positive), `teencode_path: Path`, `shopee_dir: Path = Path("raw_reviews")`, `uit_dir: Path = Path("data/uit_vsfc")`, `out_dir: Path = Path("data/clean")`. Module docstring states each default's rationale.
- Produces: `load_teencode(path: Path) -> dict[str, str]` returns the `entries` map, raising `ValueError` naming the path if `version` or `entries` is missing; `compile_teencode(table: dict[str, str]) -> Callable[[str], str]` replaces whole tokens case-insensitively.

- [ ] **Step 1: Write `config.py`** with the signature above.
- [ ] **Step 2: Write `teencode_v1.json`** with `"version": "1.0.0"` and about 100 hand-authored entries. Must include `ko`, `k`, `kh`, `hok` -> `không`; `dc`, `đc` -> `được`; `vs` -> `với`; `sp` -> `sản phẩm`; `mn` -> `mọi người`; `cx` -> `cũng`; `trc` -> `trước`; `sz` -> `size`; `j` -> `gì`; `z` -> `vậy`. Do not add `ship` or `shop`.
- [ ] **Step 3: Implement `load_teencode` and `compile_teencode`** in `teencode.py`; build one alternation regex with `(?<!\w)` / `(?!\w)` guards, longest keys first, `re.IGNORECASE`.
- [ ] **Step 4: Verify whole-token matching**

Run: `python -c "from vi_clean.config import CleanConfig; from vi_clean.teencode import load_teencode, compile_teencode; f=compile_teencode(load_teencode(CleanConfig().teencode_path)); print(f('ko biet kobe sp dc'))"`
Expected: `không biet kobe sản phẩm được`

- [ ] **Step 5: Commit** `git add vi_clean && git commit -m "feat: add CleanConfig and teencode dictionary loader"`

### Task 2: Cleaning stages

**Files:**
- Create: `vi_clean/stages.py`

**Interfaces:**
- Consumes: `CleanConfig`, `compile_teencode` from Task 1.
- Produces, all in `stages.py`, Series in and out unless noted:
  - `nfc_normalize(s: pd.Series) -> pd.Series`
  - `regex_clean(s: pd.Series, cfg: CleanConfig) -> pd.Series`
  - `expand_teencode(s: pd.Series, table: dict[str, str]) -> pd.Series`
  - `clean_texts(s: pd.Series, cfg: CleanConfig, table: dict[str, str]) -> pd.Series` runs the per-text stages in `cfg.stage_order`; non-string and null values become `""`.
  - `drop_md5_duplicates(df: pd.DataFrame, cfg: CleanConfig) -> tuple[pd.DataFrame, int]` returns the deduplicated frame (first kept) and the number of dropped rows whose label differed from the kept row's.
  - `low_ttr_mask(s: pd.Series, cfg: CleanConfig) -> pd.Series` (bool, True = spam).
  - `drop_low_ttr(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame`

- [ ] **Step 1: Implement `nfc_normalize` and `regex_clean`.** `regex_clean` removes URLs, emails, phone numbers and HTML tags, removes the Shopee template prefixes listed in `clean_reviews.py:13`, collapses runs of one character longer than `max_char_repeat`, collapses whitespace, strips, lowercases when `cfg.lowercase`; keeps emoji and tone marks.
- [ ] **Step 2: Implement `expand_teencode` and `clean_texts`** using `compile_teencode`.
- [ ] **Step 3: Implement `drop_md5_duplicates`.** Key = `hashlib.md5(...)` of casefolded text with punctuation removed and whitespace collapsed. Rows with an empty key are not deduplicated here.
- [ ] **Step 4: Implement `low_ttr_mask` and `drop_low_ttr`.** Tokens = whitespace split; flagged when token count >= `min_tokens_for_ttr` and unique/total < `ttr_min`.
- [ ] **Step 5: Verify stage behavior**

Run: `python -c "import pandas as pd; from vi_clean.config import CleanConfig; from vi_clean.teencode import load_teencode; from vi_clean.stages import *; c=CleanConfig(); t=load_teencode(c.teencode_path); s=pd.Series(['Sp quáááá tốt!!! http://x.co', 'é ko', None, '']); print(clean_texts(s,c,t).tolist()); d=pd.DataFrame({'text':['a b','A  b!','a b'],'label':['positive','negative','positive']}); print(drop_md5_duplicates(d,c)[1], low_ttr_mask(pd.Series(['ok ok ok ok ok','tốt lắm shop ơi nha']),c).tolist())"`
Expected: first list is `['sản phẩm quáá tốt!!', 'é không', '', '']` (url removed, tone mark preserved, `!!!` collapsed to `!!`); then `1 [True, False]`.

- [ ] **Step 6: Commit** `git add vi_clean/stages.py && git commit -m "feat: add five cleaning stages"`

### Task 3: Dataset merge and clean job

**Files:**
- Create: `vi_clean/merge.py`
- Modify: `.gitignore` (add `data/`)

**Interfaces:**
- Consumes: everything from Tasks 1 and 2.
- Produces: `load_uit_vsfc(cfg: CleanConfig) -> pd.DataFrame` and `load_shopee(cfg: CleanConfig) -> pd.DataFrame`, both with columns `text, label, source, rating`; `source` is `"uit_vsfc"` or `"shopee"`. Each raises `FileNotFoundError` naming the missing path. `load_shopee` reads every `cfg.shopee_dir/*.json`, takes `comment` and `rating_star` only, maps stars with `cfg.star_to_label`, and skips reviews with a missing or unmapped star.
- Produces: `run_job(cfg: CleanConfig = CleanConfig()) -> dict` merges both sources, runs `clean_texts`, drops empty text, `drop_md5_duplicates`, `drop_low_ttr`, writes `cfg.out_dir/dataset.csv` (utf-8) and `report.json`, returns the report. Report keys: `rows_in` (per source), `dropped` (`empty`, `md5_duplicate`, `low_ttr`), `label_conflicts`, `final` (total, per source, per label), `expected_total`, `difference`, `notes` (list; always includes the Fleiss-kappa sentence from the spec, and the Neutral share versus 7.4%).
- Produces: `python -m vi_clean.merge` runs `run_job` and prints the report as indented JSON.

- [ ] **Step 1: Implement the two loaders** per the signatures. UIT-VSFC reads every `cfg.uit_dir/<split>/sents.txt` with its `sentiments.txt`, pooled, mapped with `cfg.uit_label_map`; `rating` is NaN.
- [ ] **Step 2: Implement `run_job`** and the `__main__` block. Create `out_dir` if missing. The `difference` is `final.total - expected_total`, reported as is.
- [ ] **Step 3: Verify the Shopee-only path** by running with a missing UIT folder.

Run: `python -m vi_clean.merge`
Expected: `FileNotFoundError` naming `data/uit_vsfc`. Then run `python -c "from vi_clean.merge import load_shopee; from vi_clean.config import CleanConfig; d=load_shopee(CleanConfig()); print(len(d), d.label.value_counts().to_dict(), list(d.columns))"` and expect about 1000 rows, labels positive/negative/neutral, columns `['text', 'label', 'source', 'rating']`.

- [ ] **Step 4: Commit** `git add vi_clean/merge.py .gitignore && git commit -m "feat: add dataset merge-and-clean job"`

### Task 4: FastAPI app and docs

**Files:**
- Create: `vi_clean/api.py`
- Modify: `requirements.txt` (add `pandas`, `numpy`, `fastapi`, `uvicorn`), `README.md` (short section: run the job, run the API)

**Interfaces:**
- Consumes: `clean_texts`, `low_ttr_mask`, `load_teencode`, `CleanConfig`, `run_job`.
- Produces: FastAPI `app` with `POST /clean` (`{text: str}` -> `{cleaned: str, low_ttr: bool}`), `POST /clean/batch` (`{texts: list[str]}`, 1-1000 items -> `{results: [{cleaned, low_ttr}]}`), `POST /jobs/merge-clean` (no body -> report dict; a missing-data error returns HTTP 400 with the message). Config and teencode table load once at startup.

- [ ] **Step 1: Install dependencies**

Run: `pip install fastapi uvicorn`
Expected: both import without error (`python -c "import fastapi, uvicorn"`).

- [ ] **Step 2: Implement `api.py`** per the signatures.
- [ ] **Step 3: Verify endpoints**

Run: `python -c "from fastapi.testclient import TestClient; from vi_clean.api import app; c=TestClient(app); print(c.post('/clean',json={'text':'Sp quáááá ko tốt'}).json(), c.post('/clean/batch',json={'texts':['ok','']}).status_code, c.post('/clean/batch',json={'texts':[]}).status_code)"`
Expected: `{'cleaned': 'sản phẩm quáá không tốt', 'low_ttr': False} 200 422`. (TestClient needs `httpx`; if missing, run `pip install httpx` as a dev-only check and do not add it to `requirements.txt`.)

- [ ] **Step 4: Commit** `git add vi_clean/api.py requirements.txt README.md && git commit -m "feat: add FastAPI preprocessing endpoints"`

### Task 5: Real run and discrepancy note

**Files:**
- Create: `data/uit_vsfc/` (git-ignored), `docs/NOTES.md`

**Interfaces:**
- Consumes: `python -m vi_clean.merge` from Task 3.
- Produces: `docs/NOTES.md` listing every brief figure that could not be reproduced and every default chosen.

- [ ] **Step 1: Stop and confirm the download with the user.** Find the public UIT-VSFC release in the original `sents.txt`/`sentiments.txt` layout, list exact file names, sizes and URL, and ask for approval. If only parquet is available, ask before adding pyarrow.
- [ ] **Step 2: Download into `data/uit_vsfc/`** after approval and check the row count of the pooled splits against 16,175; report any difference.
- [ ] **Step 3: Run the job**

Run: `python -m vi_clean.merge`
Expected: report printed; `data/clean/dataset.csv` exists; `final.total` and `difference` shown as computed, with no filter changed.

- [ ] **Step 4: Write `docs/NOTES.md`** from the actual report: unreproduced figures (34,250, 18,075 Shopee reviews, 16,175 UIT-VSFC if different, 7.4% Neutral, Gold Set size, Fleiss' kappa, Neutral 4,982 after EDA), and the defaults from the spec.
- [ ] **Step 5: Commit** `git add docs/NOTES.md && git commit -m "docs: record unreproduced figures and chosen defaults"`
