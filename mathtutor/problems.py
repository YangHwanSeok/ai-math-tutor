import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load(name: str):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def load_problems() -> dict[str, dict]:
    return {p["id"]: p for p in _load("problems.json")}


def load_concepts() -> dict[str, dict]:
    return _load("concepts.json")
