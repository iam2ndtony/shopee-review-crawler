"""Evaluation: metrics in numpy (no scikit-learn), report assembly, latency.

Macro-F1 averages only classes present in y_true: a product with no Neutral rows must not
score a zero for a class it never had. Accuracy is reported last on purpose (it misleads
under class imbalance).
"""
import numpy as np

from .config import TrainConfig


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _prf(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    pred, true = cm.sum(axis=0), cm.sum(axis=1)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    den = prec + rec
    f1 = np.divide(2 * prec * rec, den, out=np.zeros_like(tp), where=den > 0)
    return prec, rec, f1, true


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Mean F1 over classes that have at least one true row."""
    _, _, f1, true = _prf(_confusion(y_true, y_pred, k))
    present = true > 0
    return float(f1[present].mean()) if present.any() else 0.0


def report(y_true: np.ndarray, y_pred: np.ndarray, cfg: TrainConfig) -> dict:
    """Macro-F1 (+95% bootstrap interval), per-class P/R/F1/n, confusion matrix, accuracy."""
    k = len(cfg.labels)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = _confusion(y_true, y_pred, k)
    prec, rec, f1, true = _prf(cm)
    rng = np.random.default_rng(cfg.seed)
    n = len(y_true)
    boots = [macro_f1(y_true[i], y_pred[i], k)
             for i in (rng.integers(0, n, n) for _ in range(cfg.bootstrap_n))]
    return {
        "macro_f1": round(macro_f1(y_true, y_pred, k), 4),
        "macro_f1_ci": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
        "per_class": {lab: {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                            "f1": round(float(f1[i]), 4), "n": int(true[i])}
                      for i, lab in enumerate(cfg.labels)},
        "confusion": cm.tolist(),
        "low_confidence": [lab for i, lab in enumerate(cfg.labels) if true[i] < cfg.low_conf_min],
        "accuracy": round(float((y_true == y_pred).mean()), 4),
    }
