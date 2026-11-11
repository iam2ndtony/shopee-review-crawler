"""Two-stage PhoBERT fine-tuning with leave-one-product-out evaluation.

Run: python -m vi_train.train --shopee-dir <folder with Shopee review JSON> [--run NAME
--model HF_NAME --weight-power P --no-segment]
Stage 1 trains on UIT-VSFC once; stage 2 starts from it, once per held-out Shopee product
(for out-of-fold scores) and once on all Shopee rows (the shipped model). Held-out rows are
only ever predicted, never trained on; class weights come from the stage's training rows.
Checkpoints go to checkpoints/<run>/, predictions to data/train/preds/<run>/ for evaluate.py.
"""
import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from .config import TrainConfig
from .data import class_weights, prepare, shopee_final, shopee_folds, split_uit
from .evaluate import macro_f1
from .predict import predict_probs


def train_stage(model, tokenizer, train_df: pd.DataFrame, val_df: pd.DataFrame,
                lr: float, epochs: int, cfg: TrainConfig, device: str) -> float:
    """Fine-tune in place; keep the best epoch's weights by val Macro-F1 and return that score."""
    k = len(cfg.labels)
    torch.manual_seed(cfg.seed)
    weights = torch.tensor(class_weights(train_df["y"].to_numpy(), k, cfg.weight_power),
                           dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    steps = epochs * math.ceil(len(train_df) / cfg.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_frac * steps), steps)
    texts, y = train_df["seg"].tolist(), train_df["y"].to_numpy()
    rng = np.random.default_rng(cfg.seed)
    best, best_state = -1.0, None
    model.to(device)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(texts))
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i:i + cfg.batch_size]
            enc = tokenizer([texts[j] for j in idx], truncation=True, max_length=cfg.max_len,
                            padding=True, return_tensors="pt").to(device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), torch.tensor(y[idx], device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        pred = predict_probs(model, tokenizer, val_df["seg"].tolist(), cfg.max_len,
                             cfg.batch_size, device).argmax(axis=1)
        f1 = macro_f1(val_df["y"].to_numpy(), pred, k)
        print(f"  epoch {ep + 1}/{epochs} val macro-F1 {f1:.4f}", flush=True)
        if f1 > best:
            best = f1
            best_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best


def _model(cfg: TrainConfig, device: str, state: dict | None = None):
    torch.manual_seed(cfg.seed)  # the new classifier head's random init must not differ between runs
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=len(cfg.labels))
    if state is not None:
        model.load_state_dict(state)
    return model.to(device)


def _save(model, tokenizer, path: Path, cfg: TrainConfig) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    # run.json tells inference how to feed this checkpoint (segment or not) and what it was trained with.
    (path / "run.json").write_text(json.dumps(
        {"model_name": cfg.model_name, "segment": cfg.segment, "weight_power": cfg.weight_power}), encoding="utf-8")


def _preds(model, tokenizer, df: pd.DataFrame, cfg: TrainConfig, device: str, extra=()) -> pd.DataFrame:
    probs = predict_probs(model, tokenizer, df["seg"].tolist(), cfg.max_len, cfg.batch_size, device)
    out = df[["text", "seg", "y", *extra]].copy()
    out["pred"] = probs.argmax(axis=1)
    return out


def _disjoint(*frames: pd.DataFrame) -> None:
    seen: set = set()
    for f in frames:
        idx = set(f.index)
        assert seen.isdisjoint(idx), "train/val/test rows overlap"
        seen |= idx


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shopee-dir", type=Path, default=None)
    ap.add_argument("--run", default=None, help="run name; artifacts go under checkpoints/<run>, preds/<run>, reports/<run>.json")
    ap.add_argument("--model", default=None, help="Hugging Face model name (default vinai/phobert-base)")
    ap.add_argument("--weight-power", type=float, default=None, help="class weight = (N/(K*n_c))**power (default 1.0)")
    ap.add_argument("--no-segment", action="store_true", help="skip word segmentation (for SentencePiece models like XLM-R)")
    args = ap.parse_args(argv)
    cfg = TrainConfig()
    overrides = {"shopee_dir": args.shopee_dir, "run": args.run, "model_name": args.model,
                 "weight_power": args.weight_power, "segment": False if args.no_segment else None}
    cfg = dataclasses.replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds_dir = cfg.run_preds
    preds_dir.mkdir(parents=True, exist_ok=True)
    print(f"run {cfg.run}: model {cfg.model_name}, segment {cfg.segment}, weight power {cfg.weight_power}", flush=True)

    df = prepare(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    log: dict = {"folds": {}}

    uit_tr, uit_va, uit_te = split_uit(df, cfg)
    _disjoint(uit_tr, uit_va, uit_te)
    print("stage 1: UIT-VSFC", flush=True)
    model = _model(cfg, device)
    log["stage1"] = train_stage(model, tokenizer, uit_tr, uit_va, cfg.stage1_lr, cfg.stage1_epochs, cfg, device)
    stage1_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
    _save(model, tokenizer, cfg.run_ckpt / "stage1", cfg)
    _preds(model, tokenizer, uit_te, cfg, device).to_csv(preds_dir / "uit_test_stage1.csv", index=False, encoding="utf-8-sig")

    oof = []
    for pid, tr, va, te in shopee_folds(df, cfg):
        _disjoint(tr, va, te)
        print(f"stage 2 fold, held-out product {pid}", flush=True)
        model = _model(cfg, device, stage1_state)
        log["folds"][pid] = train_stage(model, tokenizer, tr, va, cfg.stage2_lr, cfg.stage2_epochs, cfg, device)
        oof.append(_preds(model, tokenizer, te, cfg, device, extra=("product_id",)))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    pd.concat(oof).to_csv(preds_dir / "shopee_oof.csv", index=False, encoding="utf-8-sig")

    print("stage 2 final: all Shopee products", flush=True)
    tr, va = shopee_final(df, cfg)
    _disjoint(tr, va)
    model = _model(cfg, device, stage1_state)
    log["final"] = train_stage(model, tokenizer, tr, va, cfg.stage2_lr, cfg.stage2_epochs, cfg, device)
    _save(model, tokenizer, cfg.run_ckpt / "final", cfg)
    _preds(model, tokenizer, uit_te, cfg, device).to_csv(preds_dir / "uit_test_final.csv", index=False, encoding="utf-8-sig")

    with open(preds_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print("done", json.dumps(log))


if __name__ == "__main__":
    main()
