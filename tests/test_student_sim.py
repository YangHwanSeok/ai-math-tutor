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


def test_calculus_answers_in_student_notation():
    d = build_problem("power_rule_keep_exponent", 0)  # f(x)=x^3+4x^2
    assert classify(d, "f'(x)=3x^2+8x") == "correct"
    assert classify(d, "3x^3+8x^2") == "bug"
    i = build_problem("integral_no_divide", 0)  # ∫(3x^2+4x+1)dx
    assert classify(i, "x^3+2x^2+x+C") == "correct"
    assert classify(i, "F(x)=x^3+2x^2+x+5") == "correct"  # 적분상수는 어떤 값이든 인정
    assert classify(i, "3x^3+4x^2+x+C") == "bug"
    assert classify(i, "x^3+2x^2+C") == "other"


def test_calculus_correct_answer_strings():
    assert correct_answer(build_problem("integral_no_divide", 1)) == "x^3/3 - x^2 + C"
    assert correct_answer(build_problem("product_rule_naive", 0)) == "3*x^2 - 4*x + 1"
