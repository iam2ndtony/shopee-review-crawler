"""Evaluation: metrics in numpy (no scikit-learn), report assembly, latency.

Macro-F1 averages only classes present in y_true: a product with no Neutral rows must not
score a zero for a class it never had. Accuracy is reported last on purpose (it misleads
under class imbalance).
"""
import json
import time

import numpy as np
import pandas as pd

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


def _stats(times: list[float]) -> dict:
    return {"median_ms": round(float(np.median(times)), 2), "p95_ms": round(float(np.percentile(times, 95)), 2)}


def _latency(oof: pd.DataFrame, cfg: TrainConfig) -> dict:
    """Per-review latency of the shipped model on 200 Shopee reviews, after warm-up."""
    import torch
    from .data import segment
    from .predict import Classifier

    clf = Classifier.from_dir(cfg.ckpt_dir / "final", cfg)
    sample = oof.sample(n=min(200, len(oof)), random_state=cfg.seed)
    segs, raws = sample["seg"].tolist(), sample["text"].tolist()
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)

    def ms(fn) -> float:
        sync()
        t = time.perf_counter()
        fn()
        sync()
        return (time.perf_counter() - t) * 1000

    for _ in range(10):  # warm-up: CUDA kernels, allocator
        clf.probs(segs[:1])
        clf.probs(segs[:32])
    chunks = [(segs[i:i + 32], raws[i:i + 32]) for i in range(0, len(segs), 32)]
    return {
        "model_forward": {
            "batch1": _stats([ms(lambda s=s: clf.probs([s])) for s in segs]),
            "batch32": _stats([ms(lambda c=c: clf.probs(c)) / len(c) for c, _ in chunks]),
        },
        "end_to_end": {  # word segmentation + tokenize + forward
            "batch1": _stats([ms(lambda r=r: clf.probs([segment(r)])) for r in raws]),
            "batch32": _stats([ms(lambda r=r: clf.probs([segment(x) for x in r])) / len(r) for _, r in chunks]),
        },
    }


def main() -> None:
    """Build reports/metrics.json from the prediction files written by train.py."""
    cfg = TrainConfig()
    k = len(cfg.labels)
    preds = cfg.work_dir / "preds"

    def load(name: str) -> pd.DataFrame:
        return pd.read_csv(preds / name, encoding="utf-8", keep_default_na=False,
                           dtype={"text": str, "seg": str, "product_id": str})

    out: dict = {}
    for key, fname in (("uit_vsfc_test_stage1", "uit_test_stage1.csv"), ("uit_vsfc_test_final", "uit_test_final.csv")):
        d = load(fname)
        out[key] = {"n": len(d), **report(d["y"].to_numpy(), d["pred"].to_numpy(), cfg)}
    oof = load("shopee_oof.csv")
    out["shopee_oof"] = {"n": len(oof), **report(oof["y"].to_numpy(), oof["pred"].to_numpy(), cfg)}
    out["shopee_per_product"] = {
        pid: {"n": len(g), "macro_f1": round(macro_f1(g["y"].to_numpy(), g["pred"].to_numpy(), k), 4),
              "neutral_rows": int((g["y"] == 1).sum())}
        for pid, g in oof.groupby("product_id")}
    with open(preds / "train_log.json", encoding="utf-8") as f:
        out["training"] = json.load(f)
    out["latency"] = _latency(oof, cfg)
    out["notes"] = [
        "Held-out sets are not a human-labeled Gold Set; no kappa exists.",
        f"Shopee Neutral evaluation rows: {out['shopee_oof']['per_class']['neutral']['n']}; intervals are wide.",
        "Shopee labels are weak (star ratings); star-versus-text mismatches are not corrected.",
        "UIT-VSFC uses this project's own 80/10/10 split, not comparable with published results.",
    ]
    cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.report_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({key: out[key] for key in ("uit_vsfc_test_final", "shopee_oof")}, indent=2))


if __name__ == "__main__":
    main()
