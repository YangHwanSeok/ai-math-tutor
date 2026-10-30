import re
from dataclasses import dataclass
from enum import Enum

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    rationalize,
    standard_transformations,
)

x = sp.Symbol("x")

# rationalize: 소수를 정확한 분수로 읽는다 (0.1을 부동소수점으로 읽으면 1/10과 같다고 판정되지 않음).
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor, rationalize)
_UNICODE = {"²": "^2", "³": "^3", "−": "-", "×": "*", "·": "*", "÷": "/"}
_SQRT = re.compile(r"√\s*(\d+(?:\.\d+)?|\([^()]*\))")
_ROOT_SEPARATORS = re.compile(r",|;|또는|그리고|\bor\b|\band\b")
_ASSIGNMENT_CONNECTORS = re.compile(r"이고|이며|그리고|\band\b|;")
_UNITS = re.compile(r"\s*(°|도|cm\^?2|cm²|㎠|㎝|cm|개|원|점|명|살|kg)\s*$")


def _strip_units(text: str) -> str:
    return _UNITS.sub("", text.strip())


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
    misconception: bool = False  # 오개념 라이브러리와 일치함 = 우연한 실수가 아니라 일관된 잘못된 규칙

    @property
    def correct(self) -> bool:
        return self.severity is Severity.NONE


class ParseError(ValueError):
    pass


CORRECT = CheckResult(Severity.NONE, ErrorType.NONE, "정답")


def problem_symbols(problem: dict) -> dict[str, sp.Symbol]:
    """문제에서 쓰는 기호: 변수(기본 x)와 미지수(예: a, b 또는 x, y)."""
    names = [problem.get("variable", "x"), *problem.get("unknowns", [])]
    return {name: sp.Symbol(name) for name in names}


def _variable(problem: dict, symbols: dict[str, sp.Symbol]) -> sp.Symbol:
    return symbols[problem.get("variable", "x")]


def parse_math(text: str, symbols: dict[str, sp.Symbol] | None = None, evaluate: bool = True) -> sp.Expr:
    symbols = symbols or {"x": x}
    s = text.strip()
    for old, new in _UNICODE.items():
        s = s.replace(old, new)
    s = _SQRT.sub(lambda m: "sqrt" + (m.group(1) if m.group(1).startswith("(") else f"({m.group(1)})"), s)
    # parse_expr는 내부적으로 eval을 사용하므로, 허용된 기호·숫자·연산자 외의 입력은 파싱 전에 차단한다.
    # 소수점은 숫자 사이에서만 허용해 속성 접근(x.__class__ 등)을 막는다.
    letters = re.escape("".join(sorted(symbols)))
    if not s or not re.fullmatch(rf"(?:sqrt|\d+(?:\.\d+)?|[{letters}+\-*/^()\s])*", s):
        raise ParseError(text)
    try:
        return parse_expr(s, local_dict={**symbols, "sqrt": sp.sqrt}, transformations=_TRANSFORMS, evaluate=evaluate)
    except Exception as e:  # parse_expr는 잘못된 입력에 대해 여러 종류의 예외를 던진다
        raise ParseError(text) from e


def _parse_number(text: str, symbols: dict[str, sp.Symbol]) -> sp.Expr:
    value = parse_math(text, symbols)
    if value.free_symbols:
        raise ParseError(text)
    return value


def condition_equations(problem: dict, symbols: dict[str, sp.Symbol] | None = None) -> list[sp.Expr]:
    """문제의 조건을 '= 0'이어야 하는 식들의 목록으로 바꾼다."""
    symbols = symbols or problem_symbols(problem)
    cond = problem["condition"]
    if cond["type"] == "identity":
        diff = sp.expand(parse_math(cond["left"], symbols) - parse_math(cond["right"], symbols))
        return [c for c in sp.Poly(diff, _variable(problem, symbols)).all_coeffs() if c != 0]
    if cond["type"] == "equations":
        equations = []
        for eq in cond["equations"]:
            left, right = eq.split("=")
            equations.append(parse_math(left, symbols) - parse_math(right, symbols))
        return equations
    raise ValueError(f"unknown condition type: {cond['type']}")


def solve_condition(problem: dict) -> list[dict[str, sp.Expr]]:
    """조건을 만족하는 미지수 값을 SymPy로 구한다. 문제 은행에 정답을 손으로 적지 않기 위함이다."""
    symbols = problem_symbols(problem)
    unknowns = [symbols[n] for n in problem["unknowns"]]
    solutions = sp.solve(condition_equations(problem, symbols), unknowns, dict=True)
    return [{str(k): v for k, v in sol.items()} for sol in solutions]


def check_step(problem: dict, check: dict, answer: str) -> CheckResult:
    """문제(problem)의 한 풀이 단계(check)에 대한 답(answer)을 채점한다."""
    if not answer.strip():
        return CheckResult(Severity.FAR, ErrorType.NO_ANSWER, "답을 제시하지 않음")
    if check["type"] not in _CHECKERS:
        raise ValueError(f"unknown check type: {check['type']}")
    symbols = problem_symbols(problem)
    try:
        result = _CHECKERS[check["type"]](problem, check, answer, symbols)
        if result.correct:
            return result
        return _match_misconception(problem, check, answer, symbols, result) or result
    except ParseError:
        return CheckResult(Severity.UNGRADED, ErrorType.UNPARSEABLE, "수식으로 해석할 수 없는 입력")


def _match_misconception(problem, check, answer, symbols, base: CheckResult) -> CheckResult | None:
    """흔한 오개념대로 계산했을 때의 답(misconceptions)과 학생 답이 같으면 그 오개념으로 진단한다.

    오개념은 단순 실수가 아니므로 경미(MINOR)로 두지 않고 부분 오답(PARTIAL) 이상으로 올린다.
    (실험 4 1차: 부호 오개념이 '경미한 부호 실수'로 전달되자 튜터가 이번 답의 부호만 고치게 하고 규칙은 가르치지 않았다)
    """
    for m in check.get("misconceptions", []):
        if _same_answer(problem, check, symbols, answer, m["answer"]):
            if "severity" in m:
                severity = Severity[m["severity"]]
            else:
                severity = Severity.PARTIAL if base.severity is Severity.MINOR else base.severity
            error = base.error if base.error is not ErrorType.NONE else ErrorType.WRONG
            return CheckResult(severity, error, base.detail, diagnosis=m["diagnosis"], misconception=True)
    return None


def _same_answer(problem, check, symbols, answer: str, reference: str) -> bool:
    kind = check["type"]
    if kind in ("factor", "equivalent"):
        return sp.expand(parse_math(answer.split("=")[-1], symbols) - parse_math(reference, symbols)) == 0
    if kind == "roots":
        var = _variable(problem, symbols)
        a, b = _parse_roots(answer, symbols, var), _parse_roots(reference, symbols, var)
        return len(a) == len(b) and all(_contains(b, v) for v in a)
    if kind == "value":
        return _same(_parse_number(_strip_units(answer.split("=")[-1]), symbols), _parse_number(reference, symbols))
    if kind == "assignment":
        a = _parse_assignment(answer, problem["unknowns"], symbols)
        b = _parse_assignment(reference, problem["unknowns"], symbols)
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return False


def _check_factor(problem, check, answer, symbols) -> CheckResult:
    var = _variable(problem, symbols)
    target = parse_math(check.get("target", problem.get("expression", "")), symbols)
    answer = answer.split("=")[-1]
    student = parse_math(answer, symbols)
    if sp.expand(student - target) == 0:
        if _is_fully_factored(parse_math(answer, symbols, evaluate=False), var):
            return CORRECT
        return CheckResult(Severity.PARTIAL, ErrorType.INCOMPLETE, "원래 식과 같지만 완전히 인수분해되지 않음",
                           diagnosis="인수분해 미완료")
    return _compare_coefficients(target, student, var, factor_diagnosis=True)


def _check_equivalent(problem, check, answer, symbols) -> CheckResult:
    target = parse_math(check["target"], symbols)
    student = parse_math(answer.split("=")[-1], symbols)
    if sp.expand(student - target) == 0:
        return CORRECT
    return _compare_coefficients(target, student, _variable(problem, symbols))


def _check_equation(problem, check, answer, symbols) -> CheckResult:
    """활용 문제의 '식 세우기' 단계: 학생이 세운 방정식(좌변=우변)이 target=0과 같은 방정식인지 본다.

    양변에 같은 수를 곱한 식(예: 2x^2+2x-112=0)도 같은 방정식으로 인정한다.
    """
    if answer.count("=") != 1:
        return CheckResult(Severity.FAR, ErrorType.INCOMPLETE, "방정식(좌변=우변) 형태가 아님")
    left, right = answer.split("=")
    target = parse_math(check["target"], symbols)
    student = sp.expand(parse_math(left, symbols) - parse_math(right, symbols))
    if student != 0:
        ratio = sp.simplify(student / target)
        if not ratio.free_symbols and ratio != 0:
            return CORRECT
    return _compare_coefficients(target, student, _variable(problem, symbols))


def _factors(expr: sp.Expr):
    if expr.is_Mul:
        for arg in expr.args:
            yield from _factors(arg)
    elif expr.is_Pow and expr.exp.is_Integer and expr.exp > 0:
        yield from _factors(expr.base)
    else:
        yield expr


def _is_fully_factored(expr: sp.Expr, var: sp.Symbol) -> bool:
    for factor in _factors(expr):
        if not factor.has(var):
            continue
        poly = sp.Poly(factor, var)
        if poly.degree() > 1 and not poly.is_irreducible:
            return False
        if abs(poly.content()) != 1:
            return False
    return True


def _compare_coefficients(target: sp.Expr, student: sp.Expr, var: sp.Symbol,
                          factor_diagnosis: bool = False) -> CheckResult:
    expanded = sp.expand(student)
    try:
        s = sp.Poly(expanded, var)
    except sp.PolynomialError:
        return CheckResult(Severity.FAR, ErrorType.WRONG, "다항식 형태가 아님")
    t = sp.Poly(sp.expand(target), var)
    shown = f"학생 답을 전개하면 {expanded}"
    if s.degree() != t.degree():
        return CheckResult(Severity.FAR, ErrorType.WRONG, f"{shown}. 차수가 다름")

    tc, sc = t.all_coeffs(), s.all_coeffs()
    if all(_same(a, b) or _same(a, -b) for a, b in zip(tc, sc)):
        return CheckResult(Severity.MINOR, ErrorType.SIGN, f"{shown}. 계수의 크기는 같고 부호만 다름")

    diff_degrees = [t.degree() - i for i, (a, b) in enumerate(zip(tc, sc)) if not _same(a, b)]
    if len(diff_degrees) > 1:
        return CheckResult(Severity.FAR, ErrorType.WRONG, f"{shown}. 계수 {len(diff_degrees)}개가 다름")

    # x²+bx+c 꼴에서 (x+p)(x+q)의 상수항은 곱 pq, 일차항 계수는 합 p+q에 대응한다.
    monic_quadratic = factor_diagnosis and t.degree() == 2 and tc[0] == 1
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


def _parse_roots(answer: str, symbols: dict[str, sp.Symbol], var: sp.Symbol) -> list[sp.Expr]:
    roots: list[sp.Expr] = []
    for piece in _ROOT_SEPARATORS.split(re.sub(rf"{var.name}\s*=", " ", answer)):
        piece = piece.strip()
        if not piece:
            continue
        variants = [piece.replace("±", "+"), piece.replace("±", "-")] if "±" in piece else [piece]
        for v in variants:
            value = _parse_number(v, symbols)
            if not _contains(roots, value):
                roots.append(value)
    if not roots:
        raise ParseError(answer)
    return roots


def _check_roots(problem, check, answer, symbols) -> CheckResult:
    # 튜터에게 정답이 새지 않도록 detail에는 정답 값을 넣지 않는다.
    var = _variable(problem, symbols)
    target = [parse_math(r, symbols) for r in check["expected"]]
    student = _parse_roots(answer, symbols, var)
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


def _check_value(problem, check, answer, symbols) -> CheckResult:
    target = parse_math(check["expected"], symbols)
    value = _parse_number(_strip_units(answer.split("=")[-1]), symbols)
    if _same(value, target):
        return CORRECT
    if _same(value, -target):
        return CheckResult(Severity.MINOR, ErrorType.SIGN, "값의 크기는 맞지만 부호가 반대")
    return CheckResult(Severity.FAR, ErrorType.WRONG, "값이 다름")


def _parse_assignment(answer: str, unknowns: list[str], symbols: dict[str, sp.Symbol]) -> dict[str, sp.Expr]:
    """'a=5, b=3', 'a=5이고 b=3', '(x, y)=(2, 1)', '(2, 1)' 형태의 답을 {이름: 값}으로 바꾼다."""
    s = _ASSIGNMENT_CONNECTORS.sub(",", answer).strip()
    tuple_form = re.fullmatch(r"(?:\(([^()]*)\)\s*=\s*)?\(([^()]*)\)", s)
    if tuple_form:
        names = [n.strip() for n in tuple_form.group(1).split(",")] if tuple_form.group(1) else unknowns
        values = [v.strip() for v in tuple_form.group(2).split(",")]
        if len(names) != len(values):
            raise ParseError(answer)
        pairs = list(zip(names, values))
    elif "=" not in s and len(unknowns) == 1:
        pairs = [(unknowns[0], _strip_units(s))]
    else:
        pairs = []
        for piece in s.split(","):
            if not piece.strip():
                continue
            if piece.count("=") != 1:
                raise ParseError(answer)
            name, value = (p.strip() for p in piece.split("="))
            pairs.append((name, value))
    result = {}
    for name, value in pairs:
        if name not in unknowns:
            raise ParseError(answer)
        result[name] = _parse_number(value, symbols)
    if not result:
        raise ParseError(answer)
    return result


def _check_assignment(problem, check, answer, symbols) -> CheckResult:
    # 저장된 정답과 비교하지 않고 학생 값을 조건에 대입해 확인한다. 정답이 여러 개인 경우도 처리된다.
    unknowns = problem["unknowns"]
    asked = check.get("ask", unknowns)  # "a의 값을 구하시오"처럼 일부 미지수만 묻는 경우
    student = _parse_assignment(answer, asked, symbols)
    equations = condition_equations(problem, symbols)
    missing = [n for n in asked if n not in student]
    solutions = solve_condition(problem) or [{}]

    satisfied_note = ""
    if not missing and set(asked) != set(unknowns):
        if any(all(k in sol and _same(v, sol[k]) for k, v in student.items()) for sol in solutions):
            return CORRECT
    elif not missing:
        residuals = [sp.simplify(e.subs({symbols[k]: v for k, v in student.items()})) for e in equations]
        if all(r == 0 for r in residuals):
            return CORRECT
        satisfied_note = f" (조건식 {len(equations)}개 중 {sum(r == 0 for r in residuals)}개 만족)"

    best = max(solutions, key=lambda sol: sum(k in sol and _same(v, sol[k]) for k, v in student.items()))
    right = [k for k, v in student.items() if k in best and _same(v, best[k])]
    wrong = [k for k in student if k not in right]

    if not wrong:
        return CheckResult(Severity.PARTIAL, ErrorType.MISSING, f"{', '.join(missing)}의 값을 제시하지 않음")
    if not missing and all(k in best and _same(student[k], -best[k]) for k in wrong):
        return CheckResult(Severity.MINOR, ErrorType.SIGN, f"{', '.join(wrong)}의 부호가 반대{satisfied_note}")
    if right:
        return CheckResult(Severity.PARTIAL, ErrorType.WRONG,
                           f"{', '.join(right)}의 값은 맞고 {', '.join(wrong)}의 값은 틀림{satisfied_note}")
    return CheckResult(Severity.FAR, ErrorType.WRONG, f"맞는 값이 없음{satisfied_note}")


_CHECKERS = {
    "factor": _check_factor,
    "equivalent": _check_equivalent,
    "equation": _check_equation,
    "roots": _check_roots,
    "value": _check_value,
    "assignment": _check_assignment,
}
