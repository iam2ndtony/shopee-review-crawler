"""Minimal FastAPI surface for the cleaning pipeline.

Run: uvicorn vi_clean.api:app
The per-text stages run per request; MD5 duplicate detection is dataset-level and only
happens inside /jobs/merge-clean (it needs the whole dataset).
"""
import dataclasses
import os
import threading
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import CleanConfig
from .merge import run_job
from .stages import clean_texts, low_ttr_mask
from .teencode import load_teencode

cfg = CleanConfig()
table = load_teencode(cfg.teencode_path)
app = FastAPI(title="Vietnamese review cleaning")
_job_lock = threading.Lock()


class CleanIn(BaseModel):
    text: str


class BatchIn(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=1000)


def _clean(texts: list[str]) -> list[dict]:
    cleaned = clean_texts(pd.Series(texts, dtype=object), cfg, table)
    flags = low_ttr_mask(cleaned, cfg)
    return [{"cleaned": c, "low_ttr": bool(f)} for c, f in zip(cleaned, flags)]


@app.post("/clean")
def clean(body: CleanIn) -> dict:
    return _clean([body.text])[0]


@app.post("/clean/batch")
def clean_batch(body: BatchIn) -> dict:
    return {"results": _clean(body.texts)}


MAX_PREDICT_CHARS = 5000  # cleaning + word segmentation run on the full text before truncation to max_len


class PredictIn(BaseModel):
    text: str = Field(max_length=MAX_PREDICT_CHARS)


class PredictBatchIn(BaseModel):
    texts: list[Annotated[str, Field(max_length=MAX_PREDICT_CHARS)]] = Field(min_length=1, max_length=64)


_clf = None
_clf_lock = threading.Lock()


def _classifier():
    """Load the final checkpoint of run $VI_RUN once, on first use (keeps torch out of /clean's startup)."""
    global _clf
    with _clf_lock:
        if _clf is None:
            from vi_train.config import TrainConfig
            tcfg = TrainConfig()
            tcfg = dataclasses.replace(tcfg, run=os.environ.get("VI_RUN", tcfg.run))
            path = tcfg.run_ckpt / "final"
            if not path.is_dir():
                raise HTTPException(status_code=503, detail=f"Model not trained yet: checkpoints/{tcfg.run}/final is missing")
            from vi_train.predict import Classifier
            _clf = Classifier.from_dir(path, tcfg)
        return _clf


def _predict(texts: list[str]) -> list[dict]:
    # Training data was cleaned with lowercase off (PhoBERT is case-sensitive); match it here.
    cleaned = clean_texts(pd.Series(texts, dtype=object), dataclasses.replace(cfg, lowercase=False), table).tolist()
    out = [{"cleaned": c, "label": None, "probs": None} for c in cleaned]  # blank text stays null
    idx = [i for i, c in enumerate(cleaned) if c]
    if idx:  # all-blank requests never load the model
        clf = _classifier()
        from vi_train.data import segment
        inputs = [segment(cleaned[i]) if clf.segment else cleaned[i] for i in idx]  # XLM-R takes plain text
        probs = clf.probs(inputs)
        labels = clf.cfg.labels
        for i, p in zip(idx, probs):
            out[i].update(label=labels[int(p.argmax())],
                          probs={lab: round(float(x), 4) for lab, x in zip(labels, p)})
    return out


@app.post("/predict")
def predict(body: PredictIn) -> dict:
    return _predict([body.text])[0]


@app.post("/predict/batch")
def predict_batch(body: PredictBatchIn) -> dict:
    return {"results": _predict(body.texts)}


@app.post("/jobs/merge-clean")
def merge_clean() -> dict:
    """Run the full merge-and-clean job with the configured paths (no caller-supplied paths)."""
    if not _job_lock.acquire(blocking=False):  # per-process; run_job's atomic writes cover multi-worker
        raise HTTPException(status_code=409, detail="merge-clean job already running")
    try:
        return run_job(cfg)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        _job_lock.release()
