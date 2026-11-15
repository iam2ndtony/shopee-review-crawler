"""Does training on the judges' labels fix Neutral? Leave-one-product-out CV on the Gold Test Set.

Run: python -m vi_train.gold_cv --gold <3-judge CSV> [--run phobert-p0.5] [--epochs 4]
Starts from that run's UIT-VSFC stage-1 checkpoint, fine-tunes on the gold labels of the other four products
(10% of them held out to pick the best epoch) and predicts the held-out product. Every gold row is predicted
by a model that never saw its product. Writes data/train/preds/<run>/gold_cv.csv and prints Macro-F1.
"""
import argparse
import dataclasses
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from vi_clean.config import CleanConfig
from vi_clean.stages import clean_texts
from vi_clean.teencode import load_teencode

from .config import TrainConfig
from .data import segment, stratified_split
from .evaluate import macro_f1
from .train import _model, _preds, train_stage


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--run", default="phobert-p0.5", help="run whose stage-1 checkpoint and weight power are reused")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42, help="seed of the head init, batching and the validation split")
    args = ap.parse_args(argv)
    cfg = dataclasses.replace(TrainConfig(), run=args.run, weight_power=0.5, stage2_epochs=args.epochs, seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    g = pd.read_csv(args.gold, encoding="utf-8-sig")
    clean_cfg = dataclasses.replace(CleanConfig(), lowercase=False)
    g["text_c"] = clean_texts(g["text"], clean_cfg, load_teencode(clean_cfg.teencode_path))
    g = g[g["text_c"].str.strip() != ""].drop_duplicates("text_c").reset_index(drop=True)
    g["seg"] = g["text_c"].map(segment)
    g["label"] = g["majority_sentiment"]
    g["y"] = g["label"].map({lab: i for i, lab in enumerate(cfg.labels)})
    g = g.rename(columns={"text": "raw", "text_c": "text"})

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    stage1 = AutoModelForSequenceClassification.from_pretrained(cfg.run_ckpt / "stage1", num_labels=len(cfg.labels))
    state = {n: v.detach().cpu().clone() for n, v in stage1.state_dict().items()}
    del stage1

    out = []
    for product in sorted(g["product"].unique()):
        test = g[g["product"] == product]
        train, val = stratified_split(g[g["product"] != product], cfg.val_frac, cfg.seed)
        print(f"held-out product {product}: train {len(train)}, val {len(val)}, test {len(test)}", flush=True)
        model = _model(cfg, device, state)
        train_stage(model, tokenizer, train, val, cfg.stage2_lr, cfg.stage2_epochs, cfg, device)
        out.append(_preds(model, tokenizer, test, cfg, device, extra=("product", "id", "stars")))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    oof = pd.concat(out)
    oof.to_csv(cfg.run_preds / "gold_cv.csv", index=False, encoding="utf-8-sig")
    k = len(cfg.labels)
    print("gold CV macro-F1", round(macro_f1(oof["y"].to_numpy(), oof["pred"].to_numpy(), k), 4))
    for i, lab in enumerate(cfg.labels):
        tp = int(((oof["y"] == i) & (oof["pred"] == i)).sum())
        print(lab, "recall", round(tp / max(int((oof["y"] == i).sum()), 1), 3), "predicted", int((oof["pred"] == i).sum()))


if __name__ == "__main__":
    main()
