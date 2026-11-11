"""Training configuration: every hyperparameter is a named field (values from the spec).

Choices: PhoBERT-base is case-sensitive, so training data is cleaned with lowercase off;
max_len 128 covers Shopee reviews (median 16 words, p95 40); stage 2 uses a lower lr so
Shopee adapts the UIT-VSFC model instead of overwriting it.
"""
from dataclasses import dataclass
from pathlib import Path

from vi_clean.config import ROOT


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "vinai/phobert-base"
    labels: tuple[str, ...] = ("negative", "neutral", "positive")
    max_len: int = 128
    batch_size: int = 32
    stage1_lr: float = 2e-5
    stage1_epochs: int = 3
    stage2_lr: float = 1e-5
    stage2_epochs: int = 4
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    val_frac: float = 0.1
    seed: int = 42
    low_conf_min: int = 30  # classes with fewer eval rows are flagged low_confidence
    bootstrap_n: int = 1000
    shopee_dir: Path = ROOT / "raw_reviews"
    work_dir: Path = ROOT / "data" / "train"
    ckpt_dir: Path = ROOT / "checkpoints"
    reports_dir: Path = ROOT / "reports"
    # One run = one model + weight power; artifacts live under its name so runs can be compared.
    run: str = "phobert-p1.0"
    # Class weight = (N / (K * n_c)) ** weight_power. 1.0 is plain inverse frequency; lower is softer.
    weight_power: float = 1.0
    # PhoBERT needs word-segmented input; XLM-R (SentencePiece) does not.
    segment: bool = True

    @property
    def run_ckpt(self) -> Path:
        return self.ckpt_dir / self.run

    @property
    def run_preds(self) -> Path:
        return self.work_dir / "preds" / self.run

    @property
    def run_report(self) -> Path:
        return self.reports_dir / f"{self.run}.json"
