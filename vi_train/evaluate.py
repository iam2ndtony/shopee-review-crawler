"""Evaluation: metrics in numpy (no scikit-learn), report assembly, latency.

Macro-F1 averages only classes present in y_true: a product with no Neutral rows must not
score a zero for a class it never had. Accuracy is reported last on purpose (it misleads
under class imbalance). Shopee intervals resample whole products, because reviews of one
product are correlated; row-level resampling would give a misleadingly narrow interval.
"""
import argparse
import dataclasses
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


def report(y_true: np.ndarray, y_pred: np.ndarray, cfg: TrainConfig, groups: np.ndarray | None = None) -> dict:
    """Macro-F1 (+95% bootstrap interval), per-class P/R/F1/n, confusion matrix, accuracy.
    With `groups` (e.g. product ids) the bootstrap resamples whole groups instead of rows."""
    k = len(cfg.labels)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = _confusion(y_true, y_pred, k)
    prec, rec, f1, true = _prf(cm)
    rng = np.random.default_rng(cfg.seed)
    n = len(y_true)
    if groups is None:
        method = "rows"
        draws = (rng.integers(0, n, n) for _ in range(cfg.bootstrap_n))
    else:
        members = [np.flatnonzero(np.asarray(groups) == g) for g in np.unique(groups)]
        method = f"whole groups ({len(members)} products); rough with so few"
        draws = (np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
                 for _ in range(cfg.bootstrap_n))
    boots = [macro_f1(y_true[i], y_pred[i], k) for i in draws]
    return {
        "macro_f1": round(macro_f1(y_true, y_pred, k), 4),
        "macro_f1_ci": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
        "macro_f1_ci_method": method,
        "per_class": {lab: {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                            "f1": round(float(f1[i]), 4), "n": int(true[i])}
                      for i, lab in enumerate(cfg.labels)},
        "confusion": cm.tolist(),
        "low_confidence": [lab for i, lab in enumerate(cfg.labels) if true[i] < cfg.low_conf_min],
        "accuracy": round(float((y_true == y_pred).mean()), 4),
    }


def _stats(times: list[float], p95: bool = True) -> dict:
    out = {"median_ms": round(float(np.median(times)), 2)}
    if p95:
        out["p95_ms"] = round(float(np.percentile(times, 95)), 2)
    return out


def _latency(oof: pd.DataFrame, cfg: TrainConfig) -> dict:
    """Per-review latency of the shipped model on 320 Shopee reviews. Each batch is timed
    stage by stage in one loop (clean, word segmentation, tokenize+forward), so end_to_end is
    their sum on the same items; clean/segment and the model are warmed up first."""
    import torch
    from vi_clean.config import CleanConfig
    from vi_clean.stages import clean_texts
    from vi_clean.teencode import load_teencode
    from .data import segment
    from .predict import Classifier

    ccfg = dataclasses.replace(CleanConfig(), lowercase=False)
    table = load_teencode(ccfg.teencode_path)
    clf = Classifier.from_dir(cfg.run_ckpt / "final", cfg)
    seg_fn = segment if clf.segment else (lambda x: x)  # XLM-R runs take cleaned text as is
    raws = oof.sample(n=min(320, len(oof)), random_state=cfg.seed)["text"].tolist()
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)

    def timed(fn):
        sync()
        t = time.perf_counter()
        out = fn()
        sync()
        return out, (time.perf_counter() - t) * 1000

    for _ in range(10):  # warm-up: CUDA kernels, allocator, underthesea model, pandas
        clf.probs(["sản_phẩm tốt"] * 32)
        clf.probs(["sản_phẩm tốt"])
    seg_fn("sản phẩm tốt")
    clean_texts(pd.Series(["sản phẩm tốt"], dtype=object), ccfg, table)

    def run(batch: int) -> dict:
        rows = {"clean": [], "segment": [], "tokenize_forward": [], "end_to_end": []}
        for i in range(0, len(raws), batch):
            xs = raws[i:i + batch]
            cl, t1 = timed(lambda: clean_texts(pd.Series(xs, dtype=object), ccfg, table).tolist())
            sg, t2 = timed(lambda: [seg_fn(x) for x in cl])
            _, t3 = timed(lambda: clf.probs(sg))
            for key, t in zip(rows, (t1, t2, t3, t1 + t2 + t3)):
                rows[key].append(t / len(xs))  # ms per review
        full = batch == 1  # p95 over only ~10 batches would mean nothing
        return {"n_batches": len(rows["end_to_end"]), **{k_: _stats(v, p95=full) for k_, v in rows.items()}}

    return {"batch1": run(1), "batch32": run(32)}


def main(argv: list[str] | None = None) -> None:
    """Build reports/<run>.json from the prediction files written by train.py for that run."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run name (default: TrainConfig.run)")
    args = ap.parse_args(argv)
    cfg = TrainConfig()
    if args.run:
        cfg = dataclasses.replace(cfg, run=args.run)
    k = len(cfg.labels)
    preds = cfg.run_preds

    def load(name: str) -> pd.DataFrame:
        return pd.read_csv(preds / name, encoding="utf-8-sig", keep_default_na=False,
                           dtype={"text": str, "seg": str, "product_id": str})

    meta = cfg.run_ckpt / "final" / "run.json"
    out: dict = {"run": {"name": cfg.run, **json.loads(meta.read_text(encoding="utf-8"))}}
    for key, fname in (("uit_vsfc_test_stage1", "uit_test_stage1.csv"), ("uit_vsfc_test_final", "uit_test_final.csv")):
        d = load(fname)
        out[key] = {"n": len(d), **report(d["y"].to_numpy(), d["pred"].to_numpy(), cfg)}
    oof = load("shopee_oof.csv")
    out["shopee_oof"] = {"n": len(oof), **report(oof["y"].to_numpy(), oof["pred"].to_numpy(), cfg,
                                                 groups=oof["product_id"].to_numpy())}
    out["shopee_per_product"] = {
        pid: {"n": len(g), "macro_f1": round(macro_f1(g["y"].to_numpy(), g["pred"].to_numpy(), k), 4),
              "neutral_rows": int((g["y"] == 1).sum())}
        for pid, g in oof.groupby("product_id")}
    with open(preds / "train_log.json", encoding="utf-8") as f:
        out["training"] = json.load(f)
    out["latency"] = _latency(oof, cfg)
    s1, fin = out["uit_vsfc_test_stage1"]["macro_f1"], out["uit_vsfc_test_final"]["macro_f1"]
    out["notes"] = [
        "Held-out sets are not a human-labeled Gold Set; no kappa exists.",
        f"Shopee Neutral evaluation rows: {out['shopee_oof']['per_class']['neutral']['n']}; scores are rough.",
        "Shopee interval resamples whole products (5), so it is wide and itself rough; UIT-VSFC intervals resample rows.",
        "Shopee labels are weak (star ratings); star-versus-text mismatches are not corrected.",
        "UIT-VSFC uses this project's own 80/10/10 split, not comparable with published results.",
        f"Forgetting: UIT-VSFC test Macro-F1 is {s1} after stage 1 and {fin} after the Shopee stage.",
        "A product with no Neutral rows is scored on 2 classes; low_confidence is computed on pooled data only.",
        "Latency is per review on this machine's GPU, batch 1 and 32; end_to_end = clean + segment + tokenize + forward.",
    ]
    cfg.run_report.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.run_report, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({key: out[key] for key in ("uit_vsfc_test_final", "shopee_oof", "latency")}, indent=2))


if __name__ == "__main__":
    main()
