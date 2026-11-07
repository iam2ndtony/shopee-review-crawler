"""Inference helpers shared by training, evaluation and the API."""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import TrainConfig


@torch.no_grad()
def predict_probs(model, tokenizer, texts: list[str], max_len: int, batch_size: int, device: str) -> np.ndarray:
    """Softmax probabilities, shape (len(texts), num_labels); long texts are truncated."""
    model.eval()
    out = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt").to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(**enc).logits
        out[i:i + batch_size] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return out


class Classifier:
    """A saved fine-tuned checkpoint ready for inference on word-segmented text."""

    def __init__(self, model, tokenizer, cfg: TrainConfig, device: str):
        self.model, self.tokenizer, self.cfg, self.device = model, tokenizer, cfg, device

    @classmethod
    def from_dir(cls, path: Path, cfg: TrainConfig) -> "Classifier":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
        return cls(model, tokenizer, cfg, device)

    def probs(self, seg_texts: list[str]) -> np.ndarray:
        return predict_probs(self.model, self.tokenizer, seg_texts, self.cfg.max_len,
                             self.cfg.batch_size, self.device)
