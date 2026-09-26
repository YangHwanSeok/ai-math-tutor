from mathtutor.problems import load_problems
from mathtutor.solver import build_prompt, solve_with_verification

E1 = load_problems()["E1"]


def fake_llm(*responses):
    """정해진 답을 순서대로 돌려주고, 받은 메시지를 기록하는 가짜 LLM."""
    calls = []

    def complete(messages):
        calls.append(list(messages))
        return {"steps": [{"work": "", "answer": a} for a in responses[len(calls) - 1]]}

    return complete, calls


def test_correct_on_first_try():
    complete, calls = fake_llm(["(x-2)(x-3)", "2, 3"])
    record = solve_with_verification(E1, complete)
    assert record.first_try_correct and record.verified
    assert len(calls) == 1


def test_regenerates_with_feedback_after_failure():
    complete, calls = fake_llm(["(x-1)(x-6)", "1, 6"], ["(x-2)(x-3)", "2, 3"])
    record = solve_with_verification(E1, complete)
    assert not record.first_try_correct and record.verified
    assert len(record.attempts) == 2
    feedback = calls[1][-1]["content"]
    assert "1단계" in feedback and "2단계" in feedback


def test_gives_up_after_max_attempts():
    complete, calls = fake_llm(*[["(x-1)(x-6)", "1, 6"]] * 3)
    record = solve_with_verification(E1, complete, max_attempts=3)
    assert not record.verified
    assert len(calls) == 3


def test_wrong_step_count_is_rejected():
    complete, calls = fake_llm(["2, 3"], ["(x-2)(x-3)", "2, 3"])
    record = solve_with_verification(E1, complete)
    assert not record.first_try_correct and record.verified
    assert "단계 수" in calls[1][-1]["content"]


def test_prompt_lists_every_step_with_format():
    prompt = build_prompt(E1)
    assert "1. 좌변을 인수분해하기" in prompt and "2. AB=0의 성질로 해 구하기" in prompt
