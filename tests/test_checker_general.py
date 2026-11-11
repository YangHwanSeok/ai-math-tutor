import pytest
import sympy as sp

from mathtutor.checker import (
    ErrorType,
    ParseError,
    Severity,
    check_step,
    parse_math,
    problem_symbols,
    solve_condition,
)

# x^2+ax+6이 (x+2)(x+b)로 인수분해될 때 상수 a, b (정답 a=5, b=3)
U1 = {
    "variable": "x", "unknowns": ["a", "b"],
    "condition": {"type": "identity", "left": "x^2+ax+6", "right": "(x+2)(x+b)"},
}
ASSIGN = {"type": "assignment",
          "misconceptions": [{"answer": "a=6, b=3", "diagnosis": "합·곱 조건 혼동"}]}

# 연립방정식 x+y=5, x-y=1 (정답 x=3, y=2)
S1 = {"unknowns": ["x", "y"], "condition": {"type": "equations", "equations": ["x+y=5", "x-y=1"]}}


def grade(problem, check, answer):
    r = check_step(problem, check, answer)
    return r.severity, r.error, r.diagnosis


def test_condition_is_solved_by_sympy():
    assert solve_condition(U1) == [{"a": 5, "b": 3}]
    assert solve_condition(S1) == [{"x": 3, "y": 2}]


def test_unknown_letters_are_split_into_symbols():
    s = problem_symbols(U1)
    assert parse_math("x^2+ax+6", s) == s["x"] ** 2 + s["a"] * s["x"] + 6


@pytest.mark.parametrize("answer", ["a=5, b=3", "b=3, a=5", "a=5이고 b=3", "a = 5 그리고 b = 3"])
def test_assignment_correct(answer):
    assert grade(U1, ASSIGN, answer) == (Severity.NONE, ErrorType.NONE, None)


@pytest.mark.parametrize("answer, expected", [
    ("a=-5, b=3", (Severity.MINOR, ErrorType.SIGN, None)),
    ("a=6, b=3", (Severity.PARTIAL, ErrorType.WRONG, "합·곱 조건 혼동")),
    ("a=1, b=3", (Severity.PARTIAL, ErrorType.WRONG, None)),
    ("a=2, b=6", (Severity.FAR, ErrorType.WRONG, None)),
    ("a=5", (Severity.PARTIAL, ErrorType.MISSING, None)),
    ("c=5", (Severity.UNGRADED, ErrorType.UNPARSEABLE, None)),
    ("a=5 b=3", (Severity.UNGRADED, ErrorType.UNPARSEABLE, None)),
])
def test_assignment_errors(answer, expected):
    assert grade(U1, ASSIGN, answer) == expected


def test_assignment_detail_names_variables_but_not_values():
    r = check_step(U1, ASSIGN, "a=1, b=3")
    assert "b의 값은 맞고 a의 값은 틀림" in r.detail
    assert "5" not in r.detail


def test_equivalent_expansion_step():
    check = {"type": "equivalent", "target": "(x+2)(x+b)"}
    assert grade(U1, check, "x^2+(b+2)x+2b") == (Severity.NONE, ErrorType.NONE, None)
    assert grade(U1, check, "(x+2)(x+b)=x^2+bx+2x+2b") == (Severity.NONE, ErrorType.NONE, None)
    # 일차항 계수만 다르지만 인수분해 단계가 아니므로 합·곱 조건 진단은 붙이지 않는다.
    assert grade(U1, check, "x^2+bx+2b") == (Severity.PARTIAL, ErrorType.WRONG, None)


@pytest.mark.parametrize("answer", ["x=3, y=2", "(3, 2)", "(x, y)=(3, 2)", "y=2이고 x=3"])
def test_system_correct(answer):
    assert grade(S1, {"type": "assignment"}, answer)[0] is Severity.NONE


def test_system_errors():
    assert grade(S1, {"type": "assignment"}, "x=3, y=-2")[:2] == (Severity.MINOR, ErrorType.SIGN)
    r = check_step(S1, {"type": "assignment"}, "x=2, y=3")
    assert (r.severity, r.error) == (Severity.FAR, ErrorType.WRONG)
    assert "조건식 2개 중 1개 만족" in r.detail


def test_misconception_on_value_check():
    check = {"type": "value", "expected": "5",
             "misconceptions": [{"answer": "7", "diagnosis": "중앙값 정렬 누락", "severity": "PARTIAL"}]}
    assert grade({}, check, "7") == (Severity.PARTIAL, ErrorType.WRONG, "중앙값 정렬 누락")
    assert grade({}, check, "9") == (Severity.FAR, ErrorType.WRONG, None)
    assert grade({}, check, "5") == (Severity.NONE, ErrorType.NONE, None)


def test_symbols_are_limited_per_problem():
    with pytest.raises(ParseError):
        parse_math("y+1")
    with pytest.raises(ParseError):
        parse_math("a.__class__", problem_symbols(U1))
    assert parse_math("1.5x") == sp.Rational(3, 2) * sp.Symbol("x")


def test_decimals_are_exact():
    assert parse_math("0.1+0.2") == sp.Rational(3, 10)
    assert grade({}, {"type": "value", "expected": "7/2"}, "3.5")[0] is Severity.NONE


def test_equation_setup_step():
    w1 = {"variable": "x"}
    check = {"type": "equation", "target": "x^2+x-56"}
    for answer in ["x(x+1)=56", "x^2+x=56", "2x^2+2x-112=0", "56=x(x+1)"]:
        assert grade(w1, check, answer)[0] is Severity.NONE, answer
    assert grade(w1, check, "x(x+1)")[1] is ErrorType.INCOMPLETE  # 방정식이 아님
    assert grade(w1, check, "x(x+2)=56")[0] is not Severity.NONE


def test_misconception_is_never_minor():
    # 부호만 다른 답이라도 오개념 라이브러리와 일치하면 단순 실수가 아니다
    problem = {"expression": "x^2-6x+8"}
    check = {"type": "factor", "misconceptions": [{"answer": "(x+2)(x+4)", "diagnosis": "괄호 안 부호 반대"}]}
    r = check_step(problem, check, "(x+2)(x+4)")
    assert (r.severity, r.error, r.diagnosis, r.misconception) == (Severity.PARTIAL, ErrorType.SIGN, "괄호 안 부호 반대", True)
    plain = check_step(problem, {"type": "factor"}, "(x+2)(x+4)")
    assert (plain.severity, plain.misconception) == (Severity.MINOR, False)


def test_antiderivative_step():
    from mathtutor.checker import Severity, check_step
    problem = {"id": "t", "steps": []}
    check = {"type": "antiderivative", "integrand": "6x^2-4x+5"}
    assert check_step(problem, check, "2x^3-2x^2+5x+C").correct
    assert check_step(problem, check, "2x^3 - 2x^2 + 5x - 7").correct
    sign = check_step(problem, check, "2x^3+2x^2+5x+C")  # 이차항 부호만 다름 = 부호 실수
    assert sign.severity is Severity.MINOR
    coef = check_step(problem, check, "6x^3-4x^2+5x+C")  # 계수가 여러 개 다름
    assert not coef.correct and coef.severity is Severity.FAR


def test_expression_answer_leak_detection():
    from experiments.student_sim import build_problem
    from mathtutor.tutor import leaks_answer
    p = build_problem("power_rule_keep_exponent", 0)  # 정답 3x^2+8x
    assert leaks_answer("그러니까 f'(x) = 3x^2 + 8x 야", p, 0)
    assert not leaks_answer("지수를 하나 줄여 볼래?", p, 0)
