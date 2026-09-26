import pytest
import sympy as sp

from mathtutor.checker import ErrorType, ParseError, Severity, check_step, parse_math, x
from mathtutor.problems import load_concepts, load_problems

FACTOR = {"type": "factor"}
ROOTS_E1 = {"type": "roots", "expected": ["2", "3"]}


def grade(answer, check=FACTOR, expression="x^2-5x+6"):
    r = check_step(expression, check, answer)
    return r.severity, r.error


@pytest.mark.parametrize("answer", [
    "(x-2)(x-3)",
    "(x-3)(x-2)",
    "(x - 2) * (x - 3)",
    "x^2-5x+6=(x-2)(x-3)",
    "(x−2)(x−3)",
])
def test_factor_correct(answer):
    assert grade(answer) == (Severity.NONE, ErrorType.NONE)


@pytest.mark.parametrize("answer, expected", [
    ("x(x-5)+6", (Severity.PARTIAL, ErrorType.INCOMPLETE)),
    ("x^2-5x+6", (Severity.PARTIAL, ErrorType.INCOMPLETE)),
    ("(x+2)(x+3)", (Severity.MINOR, ErrorType.SIGN)),
    ("(x-1)(x-6)", (Severity.PARTIAL, ErrorType.WRONG)),
    ("(x-1)(x-4)", (Severity.PARTIAL, ErrorType.WRONG)),
    ("(x-5)(x+6)", (Severity.FAR, ErrorType.WRONG)),
    ("(x-2)", (Severity.FAR, ErrorType.WRONG)),
])
def test_factor_errors(answer, expected):
    assert grade(answer) == expected


@pytest.mark.parametrize("answer, diagnosis", [
    ("(x-1)(x-6)", "합 조건 누락"),
    ("(x-1)(x-4)", "곱 조건 누락"),
    ("x(x-5)+6", "인수분해 미완료"),
    ("(x+2)(x+3)", None),
    ("(x-5)(x+6)", None),
])
def test_factor_diagnosis(answer, diagnosis):
    assert check_step("x^2-5x+6", FACTOR, answer).diagnosis == diagnosis


def test_common_factor_must_be_fully_extracted():
    assert grade("3x(x-2)", expression="3x^2-6x") == (Severity.NONE, ErrorType.NONE)
    assert grade("x(3x-6)", expression="3x^2-6x") == (Severity.PARTIAL, ErrorType.INCOMPLETE)


def test_non_monic_one_coefficient_off_is_generic_partial():
    # (2x+1)(x+2) = 2x^2+5x+2, 목표 2x^2+7x+3 → 계수 2개 다름
    assert grade("(2x+1)(x+2)", expression="2x^2+7x+3") == (Severity.FAR, ErrorType.WRONG)
    # (2x+3)(x+1) = 2x^2+5x+3 → 일차항만 다르지만 x²+bx+c 꼴이 아니므로 합 조건으로 해석하지 않음
    r = check_step("2x^2+7x+3", FACTOR, "(2x+3)(x+1)")
    assert (r.severity, r.error, r.diagnosis) == (Severity.PARTIAL, ErrorType.WRONG, None)


@pytest.mark.parametrize("answer", ["x=2, x=3", "2 또는 3", "3,2", "x = 3 or x = 2"])
def test_roots_correct(answer):
    assert grade(answer, ROOTS_E1) == (Severity.NONE, ErrorType.NONE)


@pytest.mark.parametrize("answer, expected", [
    ("x=-2, x=-3", (Severity.MINOR, ErrorType.SIGN)),
    ("x=2, x=-3", (Severity.MINOR, ErrorType.SIGN)),
    ("x=2", (Severity.PARTIAL, ErrorType.MISSING)),
    ("x=2, x=4", (Severity.PARTIAL, ErrorType.WRONG)),
    ("5, 7", (Severity.FAR, ErrorType.WRONG)),
])
def test_roots_errors(answer, expected):
    assert grade(answer, ROOTS_E1) == expected


def test_double_root():
    assert grade("x=2", {"type": "roots", "expected": ["2"]}) == (Severity.NONE, ErrorType.NONE)


@pytest.mark.parametrize("answer", ["x=1±√2", "1+sqrt(2), 1-sqrt(2)", "x=1±√(2)"])
def test_irrational_roots(answer):
    check = {"type": "roots", "expected": ["1+sqrt(2)", "1-sqrt(2)"]}
    assert grade(answer, check) == (Severity.NONE, ErrorType.NONE)


def test_roots_detail_does_not_leak_answer():
    r = check_step("x^2-5x+6", ROOTS_E1, "x=2")
    assert "3" not in r.detail


@pytest.mark.parametrize("answer, expected", [
    ("8", (Severity.NONE, ErrorType.NONE)),
    ("b^2-4ac=8", (Severity.NONE, ErrorType.NONE)),
    ("-8", (Severity.MINOR, ErrorType.SIGN)),
    ("4", (Severity.FAR, ErrorType.WRONG)),
])
def test_value(answer, expected):
    assert grade(answer, {"type": "value", "expected": "8"}) == expected


def test_no_answer_and_unparseable():
    assert grade("") == (Severity.FAR, ErrorType.NO_ANSWER)
    assert grade("모르겠어요") == (Severity.UNGRADED, ErrorType.UNPARSEABLE)
    assert grade("(x-2)(x-") == (Severity.UNGRADED, ErrorType.UNPARSEABLE)


@pytest.mark.parametrize("payload", ["__import__('os').system('echo hi')", "exec('1')", "x.__class__"])
def test_code_injection_is_rejected(payload):
    with pytest.raises(ParseError):
        parse_math(payload)


def test_problem_bank_expected_answers_pass_checker():
    concepts = load_concepts()
    for p in load_problems().values():
        for concept in p["concepts"]:
            assert concept in concepts, (p["id"], concept)
        for step in p["steps"]:
            assert step["concept"] in concepts, (p["id"], step["concept"])
            expected = step["check"]["expected"]
            answer = ", ".join(expected) if isinstance(expected, list) else expected
            assert check_step(p["expression"], step["check"], answer).correct, (p["id"], step["goal"])


def test_problem_bank_roots_match_sympy_solution():
    for p in load_problems().values():
        solutions = set(sp.solve(parse_math(p["expression"]), x))
        for step in p["steps"]:
            if step["check"]["type"] == "roots":
                assert {parse_math(r) for r in step["check"]["expected"]} == solutions, p["id"]


def test_concept_graph_prerequisites_exist():
    concepts = load_concepts()
    for cid, c in concepts.items():
        for pre in c["prerequisites"]:
            assert pre in concepts, (cid, pre)
