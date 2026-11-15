"""Model access shared by the /predict endpoints and the web analysis.

The served run is env VI_RUN, default DEFAULT_RUN (phobert-p0.5, the power chosen on validation).
torch and the checkpoint load lazily on first use, so /clean never imports them.
"""
import dataclasses
import os
import threading

import pandas as pd

from .config import CleanConfig
from .stages import clean_texts
from .teencode import load_teencode

DEFAULT_RUN = "phobert-p0.5"

# Training data was cleaned with lowercase off (PhoBERT is case-sensitive); match it here.
_cfg = dataclasses.replace(CleanConfig(), lowercase=False)
_table = load_teencode(_cfg.teencode_path)
_clf = None
_clf_lock = threading.Lock()


class ModelMissing(RuntimeError):
    """The checkpoint of the served run does not exist."""


def serve_run() -> str:
    return os.environ.get("VI_RUN", DEFAULT_RUN)


def classifier():
    """Load the final checkpoint of the served run once."""
    global _clf
    with _clf_lock:
        if _clf is None:
            from vi_train.config import TrainConfig
            tcfg = dataclasses.replace(TrainConfig(), run=serve_run())
            path = tcfg.run_ckpt / "final"
            if not path.is_dir():
                raise ModelMissing(f"Model not trained yet: checkpoints/{tcfg.run}/final is missing")
            from vi_train.predict import Classifier
            _clf = Classifier.from_dir(path, tcfg)
        return _clf


def predict_texts(texts: list[str]) -> list[dict]:
    """Clean and classify; results keep input order. Blank text stays null; an all-blank call never loads the model."""
    cleaned = clean_texts(pd.Series(texts, dtype=object), _cfg, _table).tolist()
    out = [{"cleaned": c, "label": None, "probs": None} for c in cleaned]
    idx = [i for i, c in enumerate(cleaned) if c]
    if idx:
        clf = classifier()
        from vi_train.data import segment
        inputs = [segment(cleaned[i]) if clf.segment else cleaned[i] for i in idx]  # XLM-R takes plain text
        probs = clf.probs(inputs)
        labels = clf.cfg.labels
        for i, p in zip(idx, probs):
            out[i].update(label=labels[int(p.argmax())],
                          probs={lab: round(float(x), 4) for lab, x in zip(labels, p)})
    return out
