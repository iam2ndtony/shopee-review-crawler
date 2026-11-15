"""Classical baseline: TF-IDF word 1-2 grams with a linear softmax classifier, same leave-one-product-out folds.

Run: python -m vi_train.baseline [--run tfidf-lr]
One pooled stage (UIT-VSFC train + the four training Shopee products), class weights as in the fine-tuning
runs (power 0.5), epoch picked by validation Macro-F1. No FastText: its 7 GB vectors are not downloaded here.
Writes data/train/preds/<run>/shopee_oof.csv in the format the fine-tuned runs use.
"""
import argparse
import dataclasses
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import TrainConfig
from .data import class_weights, prepare, shopee_folds, split_uit
from .evaluate import macro_f1

MAX_FEATURES = 50_000


def grams(text: str) -> list[str]:
    w = text.lower().split()
    return w + [a + " " + b for a, b in zip(w, w[1:])]


class Tfidf:
    def __init__(self, texts):
        docs = [set(grams(t)) for t in texts]
        df = Counter(g for d in docs for g in d)
        keep = [g for g, c in df.most_common(MAX_FEATURES) if c >= 2]
        self.index = {g: i for i, g in enumerate(keep)}
        self.idf = np.array([math.log((1 + len(docs)) / (1 + df[g])) + 1 for g in keep], dtype=np.float32)

    def batch(self, texts):
        ids, offsets, vals = [], [], []
        for t in texts:
            offsets.append(len(ids))
            c = Counter(self.index[g] for g in grams(t) if g in self.index)
            v = np.array([n * self.idf[i] for i, n in c.items()], dtype=np.float32)
            ids += list(c)
            vals += (v / (np.linalg.norm(v) or 1.0)).tolist()
        return torch.tensor(ids, dtype=torch.long), torch.tensor(offsets), torch.tensor(vals)


def fit_predict(train, val, test, cfg, epochs=30, l2=1e-6):
    vec = Tfidf(train["text"].tolist())
    k = len(cfg.labels)
    torch.manual_seed(cfg.seed)
    lin = torch.nn.EmbeddingBag(len(vec.index), k, mode="sum")
    torch.nn.init.zeros_(lin.weight)
    bias = torch.nn.Parameter(torch.zeros(k))
    w = torch.tensor(class_weights(train["y"].to_numpy(), k, 0.5), dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(list(lin.parameters()) + [bias], lr=0.05, weight_decay=l2)
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
        return scores(test["text"].tolist()).argmax(1).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="tfidf-lr")
    ap.add_argument("--shopee-dir", type=Path, default=None, help="folder with the Shopee review JSON (as in train.py)")
    a = ap.parse_args(argv)
    cfg = dataclasses.replace(TrainConfig(), run=a.run, **({"shopee_dir": a.shopee_dir} if a.shopee_dir else {}))
    df = prepare(cfg)
    uit_tr, uit_va, _ = split_uit(df, cfg)
    out = []
    for pid, tr, va, te in shopee_folds(df, cfg):
        pred = fit_predict(pd.concat([uit_tr, tr]), pd.concat([uit_va, va]), te, cfg)
        o = te[["text", "seg", "y", "product_id"]].copy(); o["pred"] = pred; out.append(o)
        print(pid, "fold macro-F1", round(macro_f1(o["y"].to_numpy(), pred, len(cfg.labels)), 4), flush=True)
    oof = pd.concat(out)
    cfg.run_preds.mkdir(parents=True, exist_ok=True)
    oof.to_csv(cfg.run_preds / "shopee_oof.csv", index=False, encoding="utf-8-sig")
    print("OOF macro-F1", round(macro_f1(oof["y"].to_numpy(), oof["pred"].to_numpy(), len(cfg.labels)), 4))


if __name__ == "__main__":
    main()
