import pytest

from experiments.student_sim import MISCONCEPTIONS, build_problem, classify, correct_answer

CASES = [(mid, i) for mid, m in MISCONCEPTIONS.items() for i in range(len(m["problems"]))]


@pytest.mark.parametrize("mid,index", CASES)
def test_bug_answer_is_diagnosed_as_its_misconception(mid, index):
    problem = build_problem(mid, index)
    bug = problem["steps"][0]["check"]["misconceptions"][0]["answer"]
    assert classify(problem, bug) == "bug"


@pytest.mark.parametrize("mid,index", CASES)
def test_correct_answer_is_not_the_bug(mid, index):
    problem = build_problem(mid, index)
    assert classify(problem, correct_answer(problem)) == "correct"


def test_other_wrong_answer_and_garbage():
    problem = build_problem("root_sign", 0)  # x^2+3x+2=0, 정답 -1, -2 / 오개념 1, 2
    assert classify(problem, "3, 4") == "other"
    assert classify(problem, "모르겠어요") == "unparseable"
