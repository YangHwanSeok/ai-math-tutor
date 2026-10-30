import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 실험 1~3을 돌린 뒤 막힘 사례용으로 추가한 문제. 실험 1~3의 문제 집합(37문제)이 바뀌지 않도록 제외할 때 쓴다.
CASE_STUDY_IDS = {"W1"}


def _load(name: str):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def load_problems() -> dict[str, dict]:
    return {p["id"]: p for p in _load("problems.json")}


def load_concepts() -> dict[str, dict]:
    return _load("concepts.json")
