import json
import random
import re
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9!?'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def save_json(obj, filename: Path):
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(json.dumps(obj, indent=2), encoding="utf-8")
