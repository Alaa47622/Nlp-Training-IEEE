import json
from collections import Counter
from pathlib import Path

from .tokenize import tokenize

PAD = "<PAD>"
UNK = "<UNK>"

class Vocabulary:
    def __init__(self, max_size=50000, min_frequency=2):
        self.max_size = max_size
        self.min_frequency = min_frequency
        self.itos = [PAD, UNK]
        self.stoi = {PAD: 0, UNK: 1}

    def build(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))
        words = [w for w, n in counter.most_common() if n >= self.min_frequency]
        words = words[: self.max_size - 2]
        self.itos.extend(words)
        self.stoi.update({w: i for i, w in enumerate(self.itos)})

    def encode(self, text, max_length):
        ids = [self.stoi.get(tok, 1) for tok in tokenize(text)]
        ids = ids[:max_length]
        return ids + [0] * (max_length - len(ids))

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.itos), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        obj = cls()
        obj.itos = json.loads(path.read_text(encoding="utf-8"))
        obj.stoi = {w: i for i, w in enumerate(obj.itos)}
        return obj

    def __len__(self):
        return len(self.itos)
