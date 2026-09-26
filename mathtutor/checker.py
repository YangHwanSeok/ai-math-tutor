import re
from dataclasses import dataclass
from enum import Enum

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

x = sp.Symbol("x")

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
# parse_expr는 내부적으로 eval을 사용하므로, 수식 문자 외의 입력은 파싱 전에 차단한다.
_ALLOWED = re.compile(r"(?:sqrt|[0-9x+\-*/^().\s])*")
_UNICODE = {"²": "^2", "³": "^3", "−": "-", "×": "*", "·": "*", "÷": "/"}
_SQRT = re.compile(r"√\s*(\d+(?:\.\d+)?|\([^()]*\))")
_ROOT_SEPARATORS = re.compile(r",|;|또는|그리고|\bor\b|\band\b")


class Severity(Enum):
    NONE = "정답"
    MINOR = "경미"
    PARTIAL = "부분"
    FAR = "먼 오답"
    UNGRADED = "채점 불가"


class ErrorType(Enum):
    """단원과 무관한 일반 오류 유형. 개념별 세부 원인은 CheckResult.diagnosis에 둔다."""
    NONE = "없음"
    SIGN = "부호 실수"
    INCOMPLETE = "풀이 미완료"
    MISSING = "일부 누락"
    WRONG = "오답"
    NO_ANSWER = "무응답"
    UNPARSEABLE = "해석 불가"


@dataclass(frozen=True)
class CheckResult:
    severity: Severity
    error: ErrorType
    detail: str
    diagnosis: str | None = None

    @property
    def correct(self) -> bool:
        return self.severity is Severity.NONE


class ParseError(ValueError):
    pass


CORRECT = CheckResult(Severity.NONE, ErrorType.NONE, "정답")


def parse_math(text: str, evaluate: bool = True) -> sp.Expr:
    s = text.strip()
    for old, new in _UNICODE.items():
        s = s.replace(old, new)
    s = _SQRT.sub(lambda m: "sqrt" + (m.group(1) if m.group(1).startswith("(") else f"({m.group(1)})"), s)
    if not s or not _ALLOWED.fullmatch(s):
        raise ParseError(text)
    try:
        return parse_expr(s, local_dict={"x": x, "sqrt": sp.sqrt}, transformations=_TRANSFORMS, evaluate=evaluate)
    except Exception as e:  # parse_expr는 잘못된 입력에 대해 여러 종류의 예외를 던진다
        raise ParseError(text) from e


def check_step(expression: str, check: dict, answer: str) -> CheckResult:
    """문제 식(expression)의 한 풀이 단계(check)에 대한 답(answer)을 채점한다."""
    if not answer.strip():
        return CheckResult(Severity.FAR, ErrorType.NO_ANSWER, "답을 제시하지 않음")
    try:
        if check["type"] == "factor":
            return _check_factor(parse_math(expression), answer)
        if check["type"] == "roots":
            return _check_roots(check["expected"], answer)
        if check["type"] == "value":
            return _check_value(check["expected"], answer)
    except ParseError:
        return CheckResult(Severity.UNGRADED, ErrorType.UNPARSEABLE, "수식으로 해석할 수 없는 입력")
    raise ValueError(f"unknown check type: {check['type']}")


def _check_factor(target: sp.Expr, answer: str) -> CheckResult:
    answer = answer.split("=")[-1]
    student = parse_math(answer)
    if sp.expand(student - target) == 0:
        if _is_fully_factored(parse_math(answer, evaluate=False)):
            return CORRECT
        return CheckResult(Severity.PARTIAL, ErrorType.INCOMPLETE, "원래 식과 같지만 완전히 인수분해되지 않음",
                           diagnosis="인수분해 미완료")
    return _compare_coefficients(target, student)


def _factors(expr: sp.Expr):
    if expr.is_Mul:
        for arg in expr.args:
            yield from _factors(arg)
    elif expr.is_Pow and expr.exp.is_Integer and expr.exp > 0:
        yield from _factors(expr.base)
    else:
        yield expr


def _is_fully_factored(expr: sp.Expr) -> bool:
    for factor in _factors(expr):
        if not factor.has(x):
            continue
        poly = sp.Poly(factor, x)
        if poly.degree() > 1 and not poly.is_irreducible:
            return False
        if abs(poly.content()) != 1:
            return False
    return True


def _compare_coefficients(target: sp.Expr, student: sp.Expr) -> CheckResult:
    expanded = sp.expand(student)
    try:
        s = sp.Poly(expanded, x)
    except sp.PolynomialError:
        return CheckResult(Severity.FAR, ErrorType.WRONG, "다항식 형태가 아님")
    t = sp.Poly(target, x)
    shown = f"학생 답을 전개하면 {expanded}"
    if s.degree() != t.degree():
        return CheckResult(Severity.FAR, ErrorType.WRONG, f"{shown}. 차수가 다름")

    tc, sc = t.all_coeffs(), s.all_coeffs()
    if all(abs(a) == abs(b) for a, b in zip(tc, sc)):
        return CheckResult(Severity.MINOR, ErrorType.SIGN, f"{shown}. 계수의 크기는 같고 부호만 다름")

    diff_degrees = [t.degree() - i for i, (a, b) in enumerate(zip(tc, sc)) if a != b]
    if len(diff_degrees) > 1:
        return CheckResult(Severity.FAR, ErrorType.WRONG, f"{shown}. 계수 {len(diff_degrees)}개가 다름")

    # x²+bx+c 꼴에서 (x+p)(x+q)의 상수항은 곱 pq, 일차항 계수는 합 p+q에 대응한다.
    monic_quadratic = t.degree() == 2 and tc[0] == 1
    if monic_quadratic and diff_degrees[0] == 0:
        return CheckResult(Severity.PARTIAL, ErrorType.WRONG,
                           f"{shown}. 상수항만 다름 (합 조건은 만족, 곱 조건 불만족)", diagnosis="곱 조건 누락")
    if monic_quadratic and diff_degrees[0] == 1:
        return CheckResult(Severity.PARTIAL, ErrorType.WRONG,
                           f"{shown}. 일차항 계수만 다름 (곱 조건은 만족, 합 조건 불만족)", diagnosis="합 조건 누락")
    return CheckResult(Severity.PARTIAL, ErrorType.WRONG, f"{shown}. {diff_degrees[0]}차항 계수만 다름")


def _same(a: sp.Expr, b: sp.Expr) -> bool:
    return sp.simplify(a - b) == 0


def _contains(values: list[sp.Expr], v: sp.Expr) -> bool:
    return any(_same(v, w) for w in values)


def _parse_roots(answer: str) -> list[sp.Expr]:
    roots: list[sp.Expr] = []
    for piece in _ROOT_SEPARATORS.split(re.sub(r"x\s*=", " ", answer)):
        piece = piece.strip()
        if not piece:
            continue
        variants = [piece.replace("±", "+"), piece.replace("±", "-")] if "±" in piece else [piece]
        for v in variants:
            value = parse_math(v)
            if value.has(x):
                raise ParseError(answer)
            if not _contains(roots, value):
                roots.append(value)
    if not roots:
        raise ParseError(answer)
    return roots


def _check_roots(expected: list[str], answer: str) -> CheckResult:
    # 튜터에게 정답이 새지 않도록 detail에는 정답 값을 넣지 않는다.
    target = [parse_math(r) for r in expected]
    student = _parse_roots(answer)
    matched = [s for s in student if _contains(target, s)]
    if len(matched) == len(student) == len(target):
        return CORRECT
    if len(student) == len(target) and all(_contains(target, s) or _contains(target, -s) for s in student):
        return CheckResult(Severity.MINOR, ErrorType.SIGN, "해의 크기는 맞지만 부호가 반대인 해가 있음")
    if matched and len(matched) == len(student):
        return CheckResult(Severity.PARTIAL, ErrorType.MISSING, f"해 {len(target)}개 중 {len(student)}개만 제시")
    if matched:
        return CheckResult(Severity.PARTIAL, ErrorType.WRONG, f"제시한 해 {len(student)}개 중 {len(matched)}개만 맞음")
    return CheckResult(Severity.FAR, ErrorType.WRONG, "맞는 해가 없음")


def _check_value(expected: str, answer: str) -> CheckResult:
    target = parse_math(expected)
    value = parse_math(answer.split("=")[-1])
    if value.has(x):
        raise ParseError(answer)
    if _same(value, target):
        return CORRECT
    if _same(value, -target):
        return CheckResult(Severity.MINOR, ErrorType.SIGN, "값의 크기는 맞지만 부호가 반대")
    return CheckResult(Severity.FAR, ErrorType.WRONG, "값이 다름")
