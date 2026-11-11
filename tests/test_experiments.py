"""실험 스크립트 중 API를 쓰지 않는 부분."""
import json
from types import SimpleNamespace

from experiments.exp4_tutoring import BRIDGE_DECISION_SCHEMA, BRIDGE_RESPONSE_PROMPT, reuse_pres
from experiments.exp5_diagnosis import build_items, diagnose_constraint, diagnose_library
from experiments.student_sim import MISCONCEPTIONS


def test_bridge_prompt_accepts_decision():
    decision = {"reason": "r", "error_type": "careless", "strategy": "Provide a hint", "intention": "Hint at the mistake"}
    assert set(BRIDGE_DECISION_SCHEMA["required"]) == set(decision)
    assert "Provide a hint" in BRIDGE_RESPONSE_PROMPT.format(**decision)


def test_reuse_pres_rebuilds_history(tmp_path):
    pre = {"rep": 0, "misconception": "root_sign", "index": 0, "p1": "root_sign-0",
           "pre_reply": {"message": "괄호 안 수가 해예요", "answer": "1, 2"}, "pre_label": "bug"}
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"meta": {"first_turn": "answer", "misconceptions": ["root_sign"]}, "pre": [pre]}),
                 encoding="utf-8")
    args = SimpleNamespace(first_turn="answer", misconceptions=None)
    [p] = reuse_pres(f, args)
    assert args.misconceptions == ["root_sign"]
    assert p["history"][0]["role"] == "user" and p["history"][1]["content"].endswith("답: 1, 2")


def test_diagnosis_items_library_method():
    items = build_items()
    bugs = [it for it in items if it["kind"] == "bug"]
    assert len(bugs) == 6 * len(MISCONCEPTIONS)
    assert all(diagnose_library(it)["pred"] == it["truth"] for it in bugs)
    unseen = [it for it in items if it["kind"] == "unseen"]
    assert unseen and all(diagnose_library(it)["pred"] == "none" for it in unseen)


def test_constraint_method_cannot_name_misconceptions():
    items = build_items(only=["root_sign"])
    bug = next(it for it in items if it["kind"] == "bug")
    assert diagnose_constraint(bug)["severity"] == "MINOR"  # 해의 부호 반대는 '부호 실수'로만 보인다
