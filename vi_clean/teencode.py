"""Teencode dictionary loader. The dictionary is a hand-curated, versioned JSON file
(never derived from the dataset); matching is whole-token and case-insensitive."""
import json
import re
from pathlib import Path
from typing import Callable


def load_teencode(path: Path) -> dict[str, str]:
    """Return the `entries` map; fail loudly if the file is not a valid dictionary."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "version" not in data or "entries" not in data:
        raise ValueError(f"{path}: teencode file needs 'version' and 'entries'")
    return {k.lower(): v for k, v in data["entries"].items()}


def compile_teencode(table: dict[str, str]) -> Callable[[str], str]:
    """Build a one-pass replacer; (?<!\\w)/(?!\\w) keep 'ko' from touching 'kobe'."""
    if not table:
        return lambda s: s
    keys = sorted(table, key=len, reverse=True)
    pat = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, keys)) + r")(?!\w)", re.IGNORECASE)
    return lambda s: pat.sub(lambda m: table[m.group(1).lower()], s)
