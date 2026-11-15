"""Classical baseline: TF-IDF word n-grams with a linear softmax classifier, same leave-one-product-out folds.

Run: python -m vi_train.baseline --shopee-dir <folder> [--tune] [--seeds 42 43 44]
One pooled stage (UIT-VSFC train + the training Shopee products), class weights as in the fine-tuning runs
(power 0.5), epoch picked by validation Macro-F1. --tune first searches n-gram range, feature cap, L2 and
learning rate on one validation split (UIT-VSFC validation + a stratified tenth of Shopee), never on held-out
products, then runs every fold with the best setting. No FastText: its 7 GB vectors are not downloaded here.
Writes data/train/preds/<run>-s<seed>/shopee_oof.csv in the format the fine-tuned runs use.
"""
import argparse
import dataclasses
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import TrainConfig
from .data import class_weights, prepare, shopee_final, shopee_folds, split_uit
from .evaluate import macro_f1

DEFAULT_HP = {"ngram": 2, "max_features": 50_000, "l2": 1e-6, "lr": 0.05}
GRID = {"ngram": [1, 2], "max_features": [20_000, 50_000], "l2": [1e-7, 1e-6, 1e-5], "lr": [0.02, 0.05, 0.1]}


def grams(text: str, ngram: int) -> list[str]:
    w = text.lower().split()
    return w + ([a + " " + b for a, b in zip(w, w[1:])] if ngram == 2 else [])


class Tfidf:
    def __init__(self, texts, ngram, max_features):
        self.ngram = ngram
        docs = [set(grams(t, ngram)) for t in texts]
        df = Counter(g for d in docs for g in d)
        keep = [g for g, c in df.most_common(max_features) if c >= 2]
        self.index = {g: i for i, g in enumerate(keep)}
        self.idf = np.array([math.log((1 + len(docs)) / (1 + df[g])) + 1 for g in keep], dtype=np.float32)

    def batch(self, texts):
        ids, offsets, vals = [], [], []
        for t in texts:
            offsets.append(len(ids))
            c = Counter(self.index[g] for g in grams(t, self.ngram) if g in self.index)
            v = np.array([n * self.idf[i] for i, n in c.items()], dtype=np.float32)
            ids += list(c)
            vals += (v / (np.linalg.norm(v) or 1.0)).tolist()
        return torch.tensor(ids, dtype=torch.long), torch.tensor(offsets), torch.tensor(vals)


def fit_predict(train, val, test, cfg, hp, epochs=30):
    """Train on `train`, keep the epoch with the best `val` Macro-F1; return (that score, predictions for `test`)."""
    vec = Tfidf(train["text"].tolist(), hp["ngram"], hp["max_features"])
    k = len(cfg.labels)
    torch.manual_seed(cfg.seed)
    lin = torch.nn.EmbeddingBag(len(vec.index), k, mode="sum")
    torch.nn.init.zeros_(lin.weight)
    bias = torch.nn.Parameter(torch.zeros(k))
    w = torch.tensor(class_weights(train["y"].to_numpy(), k, cfg.weight_power), dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(list(lin.parameters()) + [bias], lr=hp["lr"], weight_decay=hp["l2"])
    texts, y = train["text"].tolist(), torch.tensor(train["y"].to_numpy())
    rng = np.random.default_rng(cfg.seed)

    def scores(ts):
        ids, off, vals = vec.batch(ts)
        return lin(ids, off, per_sample_weights=vals) + bias

    best, best_state = -1.0, None
    for _ in range(epochs):
        order = rng.permutation(len(texts))
        for i in range(0, len(order), 256):
            idx = order[i:i + 256]
            loss = loss_fn(scores([texts[j] for j in idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            f1 = macro_f1(val["y"].to_numpy(), scores(val["text"].tolist()).argmax(1).numpy(), k)
        if f1 > best:
            best, best_state = f1, ({n: p.detach().clone() for n, p in lin.state_dict().items()}, bias.detach().clone())
    lin.load_state_dict(best_state[0]); bias.data = best_state[1]
    with torch.no_grad():
        return best, scores(test["text"].tolist()).argmax(1).numpy()


def tune(df, cfg) -> dict:
    """Grid search on one validation split: UIT-VSFC validation plus a stratified tenth of Shopee."""
    uit_tr, uit_va, _ = split_uit(df, cfg)
    s_tr, s_va = shopee_final(df, cfg)
    train, val = pd.concat([uit_tr, s_tr]), pd.concat([uit_va, s_va])
    results = []
    for vals in itertools.product(*GRID.values()):
        hp = dict(zip(GRID, vals))
        score, _ = fit_predict(train, val, val.iloc[:1], cfg, hp)
        results.append({**hp, "val_macro_f1": round(float(score), 4)})
        print(results[-1], flush=True)
    return max(results, key=lambda r: r["val_macro_f1"]) | {"grid": results}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="tfidf-lr")
    ap.add_argument("--shopee-dir", type=Path, default=None, help="folder with the Shopee review JSON (as in train.py)")
    ap.add_argument("--tune", action="store_true", help="search the grid on validation data first")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    a = ap.parse_args(argv)
    base = dataclasses.replace(TrainConfig(), run=a.run, weight_power=0.5, segment=False,
                               **({"shopee_dir": a.shopee_dir} if a.shopee_dir else {}))
    df = prepare(base)
    hp = dict(DEFAULT_HP)
    if a.tune:
        found = tune(df, base)
        hp = {k: found[k] for k in GRID}
        (base.work_dir / "preds").mkdir(parents=True, exist_ok=True)
        (base.work_dir / "preds" / f"{a.run}_tuning.json").write_text(json.dumps(found, indent=2), encoding="utf-8")
    print("hyperparameters", hp, flush=True)
    for seed in a.seeds:
        cfg = dataclasses.replace(base, seed=seed, run=f"{a.run}-s{seed}")
        uit_tr, uit_va, _ = split_uit(df, cfg)
        out = []
        for pid, tr, va, te in shopee_folds(df, cfg):
            _, pred = fit_predict(pd.concat([uit_tr, tr]), pd.concat([uit_va, va]), te, cfg, hp)
            o = te[["text", "seg", "y", "product_id"]].copy(); o["pred"] = pred; out.append(o)
        oof = pd.concat(out)
        cfg.run_preds.mkdir(parents=True, exist_ok=True)
        oof.to_csv(cfg.run_preds / "shopee_oof.csv", index=False, encoding="utf-8-sig")
        print(f"seed {seed} OOF macro-F1", round(macro_f1(oof["y"].to_numpy(), oof["pred"].to_numpy(), len(cfg.labels)), 4), flush=True)


if __name__ == "__main__":
    main()
