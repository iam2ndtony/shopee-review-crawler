"""Pipeline configuration. Every open parameter of the brief is a named field here.

Defaults and why:
- stage_order: NFC first so decomposed Vietnamese cannot defeat later regexes;
  teencode runs before dedup/TTR so "ko" and "không" count as the same text.
  (The brief lists regex, NFC, ..., teencode last; this is a deliberate change.)
- lowercase: Vietnamese sentiment cues are case-insensitive; shrinks the vocabulary.
- max_char_repeat: "quáááá" -> "quáá" keeps the emphasis signal, drops the noise.
- min_tokens_for_ttr / ttr_min: TTR is meaningless on very short texts, so texts under
  5 tokens are exempt; below 0.3 unique/total is treated as repetitive bot text.
- star_to_label: Shopee stars are weak labels (1-2 neg, 3 neu, 4-5 pos).
- uit_label_map: UIT-VSFC sentiment ids 0/1/2 = negative/neutral/positive.
- expected_total: the brief's figure, only ever compared against, never targeted.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # anchored so cwd does not matter


@dataclass(frozen=True)
class CleanConfig:
    stage_order: tuple[str, ...] = ("nfc", "regex", "teencode")
    lowercase: bool = True
    max_char_repeat: int = 2
    min_tokens_for_ttr: int = 5
    ttr_min: float = 0.3
    expected_total: int = 34250
    star_to_label: dict[int, str] = field(default_factory=lambda: {
        1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"})
    uit_label_map: dict[int, str] = field(default_factory=lambda: {
        0: "negative", 1: "neutral", 2: "positive"})
    teencode_path: Path = ROOT / "vi_clean" / "data" / "teencode_v1.json"
    shopee_dir: Path = ROOT / "raw_reviews"
    uit_dir: Path = ROOT / "data" / "uit_vsfc"
    out_dir: Path = ROOT / "data" / "clean"
