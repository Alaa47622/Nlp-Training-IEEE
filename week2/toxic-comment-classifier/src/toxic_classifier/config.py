from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def path(name: str) -> Path:
    return ROOT / name
