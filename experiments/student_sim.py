"""오개념을 가진 가상 학생.

오개념마다 (1) 학생이 믿는 방법, (2) 그 방법을 겨냥한 모범 설명, (3) 시험 문제를 둔다.
시험 문제의 정답과 '오개념대로 풀었을 때의 답'은 SymPy로 계산한다(BUGGY의 버그 라이브러리와 같은 발상).
학생 답이 오개념 답과 같으면 그 오개념이 나타난 것으로 판정한다.
"""
import sympy as sp

from mathtutor import llm
from mathtutor.checker import ErrorType, check_step, parse_math, x
from mathtutor.solver import ANSWER_FORMATS


def _factor_sign(expr: sp.Expr) -> str:
    # 두 수 p, q는 맞게 찾고 (x-p)(x-q)로 씀 = x 자리에 -x를 넣은 식
    return str(sp.factor(sp.expand(expr.subs(x, -x))))


def _roots(left: str, right: str) -> list[sp.Expr]:
    return sp.solve(parse_math(left) - parse_math(right), x)


def _root_sign(left: str, right: str) -> list[sp.Expr]:
    return [-r for r in _roots(left, right)]


def _divide_by_x(left: str, right: str) -> list[sp.Expr]:
    return [r for r in _roots(left, right) if r != 0]


def _positive_sqrt_only(left: str, right: str) -> list[sp.Expr]:
    # (x-a)^2=c 에서 x-a=+sqrt(c)만 택하면 항상 큰 쪽 해가 된다
    return [max(_roots(left, right))]


def _each_factor_equals_right(left: str, right: str) -> list[sp.Expr]:
    # (x-a)(x-b)=k 에서 x-a=k, x-b=k로 놓음
    k = parse_math(right)
    factors = [f for f in sp.Mul.make_args(parse_math(left)) if f.has(x)]
    return [sp.solve(f - k, x)[0] for f in factors]


# 통계·도형: 답이 수 하나인 문제. 문제마다 매개변수를 두고 정답과 오개념 답을 계산한다.
def _data(values: list[int]) -> str:
    return ", ".join(map(str, values))


def _median(values: list[int]) -> sp.Expr:
    v = sorted(values)
    n = len(v)
    return sp.Integer(v[n // 2]) if n % 2 else sp.Rational(v[n // 2 - 1] + v[n // 2], 2)


def _mode(values: list[int]) -> int:
    return max(set(values), key=values.count)


def _squared_deviation_sum(values: list[int]) -> sp.Expr:
    mean = sp.Rational(sum(values), len(values))
    return sum((v - mean) ** 2 for v in values)


def _variance(values: list[int]) -> sp.Expr:
    return _squared_deviation_sum(values) / len(values)


MISCONCEPTIONS = {
    "factor_sign": {
        "name": "인수분해 부호 반대",
        "kind": "factor",
        "concept": "factor_trinomial",
        "diagnosis": "두 수는 맞게 찾았지만 괄호 안에 부호를 반대로 씀",
        "belief": "x^2+bx+c를 인수분해할 때 더해서 b, 곱해서 c가 되는 두 수 p, q를 찾은 다음 (x-p)(x-q)로 쓴다.",
        "example": "x^2+7x+10 → 더해서 7, 곱해서 10인 수는 2와 5 → (x-2)(x-5)",
        "explanation": "더해서 7, 곱해서 10인 두 수 2와 5를 찾았다면 (x+2)(x+5)로 써야 해. "
                       "(x+2)(x+5)를 전개하면 x^2+7x+10이지만, (x-2)(x-5)를 전개하면 x^2-7x+10이 돼서 일차항 부호가 달라져. "
                       "찾은 두 수를 부호 그대로 괄호 안에 넣고, 마지막에 전개해서 원래 식과 같은지 꼭 확인하자.",
        "problems": ["x^2+7x+12", "x^2-6x+8", "x^2+2x-15", "x^2-x-12", "x^2+9x+20", "x^2-8x+15"],
        "bug": _factor_sign,
    },
    "root_sign": {
        "name": "해의 부호 반대",
        "kind": "roots",
        "concept": "zero_product",
        "diagnosis": "인수 x+p의 p를 그대로 해로 씀 (x+p=0을 풀지 않음)",
        "belief": "인수분해해서 (x+p)(x+q)=0이 되면 괄호 안에서 x 뒤에 붙은 수 p, q를 부호까지 그대로 쓴 것이 해이다.",
        "example": "x^2+5x+4=0 → (x+1)(x+4)=0 → 해는 1, 4 / x^2-5x+6=0 → (x-2)(x-3)=0 → 해는 -2, -3",
        "explanation": "(x+1)(x+4)=0이면 x+1=0 또는 x+4=0이어야 해. 그러니까 해는 x=-1 또는 x=-4야. "
                       "괄호 안의 수가 해가 아니라, 괄호 하나를 0으로 만드는 x가 해야. "
                       "x=1을 원래 식에 넣어 보면 1+5+4=10이라 0이 안 되지? 구한 해는 대입해서 0이 되는지 확인하자.",
        "problems": ["x^2+3x+2=0", "x^2-7x+10=0", "x^2+x-6=0", "x^2-2x-8=0", "x^2+8x+15=0", "x^2-9x+14=0"],
        "bug": _root_sign,
    },
    "divide_by_x": {
        "name": "양변을 x로 나눔",
        "kind": "roots",
        "concept": "quad_by_factoring",
        "diagnosis": "양변을 x로 나누어 x=0인 해를 잃음",
        "belief": "x가 들어 있는 식은 양변을 x로 나누어 간단히 만든 뒤 푼다.",
        "example": "x^2=3x → 양변을 x로 나누면 x=3 → 해는 3",
        "explanation": "x^2=3x에서 양변을 x로 나누면 안 돼. x가 0일 수도 있는데, 0으로는 나눌 수 없거든. "
                       "대신 x^2-3x=0으로 한쪽에 모으고 x(x-3)=0으로 인수분해하면 x=0 또는 x=3, 해가 두 개야. "
                       "x=0을 넣어 봐도 0=0으로 성립하지. x로 나누지 말고 묶어서 인수분해하자.",
        "problems": ["x^2=7x", "x^2-6x=0", "2x^2=10x", "x^2+5x=0", "4x^2=12x", "x^2=-3x"],
        "bug": _divide_by_x,
    },
    "positive_sqrt_only": {
        "name": "제곱근의 양수만 택함",
        "kind": "roots",
        "concept": "quad_by_sqrt",
        "diagnosis": "제곱근을 구할 때 ±를 빠뜨려 음수 쪽 해를 잃음",
        "belief": "x^2=9이면 제곱해서 9가 되는 수는 3이므로 x=3이다. (x-a)^2=c도 x-a=sqrt(c)로 푼다.",
        "example": "x^2=25 → x=5 / (x-1)^2=4 → x-1=2 → x=3",
        "explanation": "제곱해서 25가 되는 수는 5만이 아니야. (-5)^2=25이기도 하니까 x=5 또는 x=-5, 즉 x=±5야. "
                       "(x-1)^2=4라면 x-1=2 또는 x-1=-2, 두 경우를 모두 풀어서 x=3 또는 x=-1이 돼. "
                       "제곱근을 구할 때는 항상 ±를 붙이자.",
        "problems": ["x^2=16", "x^2-49=0", "(x-2)^2=9", "(x+1)^2=4", "2x^2=18", "(x-3)^2=25"],
        "bug": _positive_sqrt_only,
    },
    "nonzero_product": {
        "name": "곱이 0이 아닐 때 인수마다 우변과 같게 놓음",
        "kind": "roots",
        "concept": "zero_product",
        "diagnosis": "곱이 0이 아닌데 각 인수를 우변과 같게 놓음 (AB=0의 성질을 잘못 적용)",
        "belief": "곱셈 꼴로 된 방정식은 각 괄호를 우변과 같게 놓고 푼다.",
        "example": "(x-2)(x-3)=6 → x-2=6 또는 x-3=6 → 해는 8, 9",
        "explanation": "A×B=0일 때만 A=0 또는 B=0이라고 할 수 있어. 곱이 0이면 둘 중 하나는 반드시 0이어야 하니까. "
                       "그런데 (x-1)(x-3)=8처럼 곱이 8이면 1×8, 2×4, 4×2처럼 경우가 너무 많아서 x-1=8이라고 놓을 수 없어. "
                       "먼저 전개해서 우변을 0으로 만들자: x^2-4x+3-8=0, x^2-4x-5=0, (x-5)(x+1)=0이니까 x=5 또는 x=-1이야.",
        "problems": ["(x-1)(x-2)=6", "x(x-1)=12", "x(x+2)=8", "(x+1)(x+2)=12", "(x-4)(x-1)=10", "x(x-3)=10"],
        "bug": _each_factor_equals_right,
    },
    "median_unsorted": {
        "name": "정렬하지 않고 가운데 값을 고름",
        "kind": "value",
        "concept": "median",
        "diagnosis": "자료를 크기순으로 정렬하지 않고 주어진 순서의 가운데 값을 고름",
        "belief": "중앙값은 자료에서 가운데 자리에 있는 값이다. 주어진 순서 그대로 가운데 자리의 값을 고른다.",
        "example": "3, 8, 1, 5, 9 → 가운데 자리의 값은 1 → 중앙값 1",
        "explanation": "중앙값은 자료를 작은 값부터 크기순으로 늘어놓았을 때 가운데 오는 값이야. "
                       "3, 8, 1, 5, 9를 정렬하면 1, 3, 5, 8, 9니까 중앙값은 1이 아니라 5야. "
                       "주어진 순서는 아무 의미가 없으니까 항상 먼저 정렬하자.",
        "problems": [[6, 1, 9, 3, 7], [4, 10, 2, 8, 5], [12, 5, 3, 9, 7], [7, 15, 2, 11, 4, 9, 6], [3, 9, 1, 6, 8],
                     [10, 4, 13, 2, 7]],
        "statement": lambda v: f"다음 자료의 중앙값을 구하시오: {_data(v)}",
        "answer": _median,
        "bug": lambda v: sp.Integer(v[len(v) // 2]),
    },
    "mode_frequency": {
        "name": "최빈값과 도수 혼동",
        "kind": "value",
        "concept": "mode",
        "diagnosis": "가장 많이 나온 값이 아니라 그 값이 나온 횟수(도수)를 최빈값으로 씀",
        "belief": "최빈값은 가장 많이 나온 횟수이다.",
        "example": "2, 6, 6, 1 → 6이 2번 나왔으니 최빈값은 2",
        "explanation": "최빈값은 가장 많이 나온 '값' 자체야. 횟수가 아니야. "
                       "2, 6, 6, 1에서 가장 많이 나온 값은 6이니까 최빈값은 6이고, 2는 6이 나온 횟수(도수)야. "
                       "횟수는 어떤 값을 고를지 정할 때만 쓰고, 답은 그 값을 쓰자.",
        "problems": [[1, 4, 4, 6, 4, 2], [7, 7, 2, 9, 7, 7, 1], [3, 8, 8, 1, 5], [6, 2, 6, 9, 6, 4], [9, 1, 9, 3, 9, 3],
                     [5, 10, 10, 10, 2, 10, 10]],
        "statement": lambda v: f"다음 자료의 최빈값을 구하시오: {_data(v)}",
        "answer": lambda v: sp.Integer(_mode(v)),
        "bug": lambda v: sp.Integer(v.count(_mode(v))),
    },
    "variance_no_divide": {
        "name": "분산에서 개수로 나누지 않음",
        "kind": "value",
        "concept": "variance",
        "diagnosis": "편차 제곱의 합을 자료의 개수로 나누지 않음",
        "belief": "분산은 각 편차를 제곱해서 모두 더한 값이다.",
        "example": "1, 3, 5 → 평균 3, 편차 -2, 0, 2 → 4+0+4=8 → 분산 8",
        "explanation": "분산은 편차 제곱의 '평균'이야. 그래서 편차를 제곱해서 더한 다음 자료의 개수로 나눠야 해. "
                       "1, 3, 5라면 편차 제곱의 합이 8이고 자료가 3개니까 분산은 8/3이야. "
                       "합만 구하면 자료가 많을수록 값이 커지기만 해서 흩어진 정도를 비교할 수 없거든.",
        "problems": [[1, 2, 3, 4, 5], [2, 4, 4, 6], [3, 5, 7, 9], [4, 6, 8, 10, 12], [2, 3, 7, 8], [1, 4, 7]],
        "statement": lambda v: f"다음 자료의 분산을 구하시오: {_data(v)}",
        "answer": _variance,
        "bug": _squared_deviation_sum,
    },
    "std_no_sqrt": {
        "name": "표준편차를 분산으로 답함",
        "kind": "value",
        "concept": "variance",
        "diagnosis": "분산까지 구하고 제곱근을 씌우지 않음 (분산과 표준편차 혼동)",
        "belief": "표준편차는 편차를 제곱해서 평균 낸 값이다.",
        "example": "1, 3, 5 → 편차 제곱 4, 0, 4 → 평균 8/3 → 표준편차 8/3",
        "explanation": "편차를 제곱해서 평균 낸 값은 분산이야. 표준편차는 분산의 양의 제곱근이야. "
                       "1, 3, 5라면 분산이 8/3이니까 표준편차는 sqrt(8/3)이야. "
                       "제곱했던 단위를 원래대로 돌려놓으려고 제곱근을 씌운다고 생각하면 돼.",
        "problems": [[2, 4, 4, 4, 5, 5, 7, 9], [1, 2, 3, 4, 5], [3, 5, 7, 9], [1, 4, 7], [2, 3, 7, 8],
                     [4, 6, 8, 10, 12]],
        "statement": lambda v: f"다음 자료의 표준편차를 구하시오: {_data(v)}",
        "answer": lambda v: sp.sqrt(_variance(v)),
        "bug": _variance,
    },
    "polygon_angle_n": {
        "name": "다각형 내각의 합을 180°×n으로 계산",
        "kind": "value",
        "concept": "polygon_angle",
        "diagnosis": "n각형의 내각의 합을 180°×(n-2)가 아니라 180°×n으로 계산",
        "belief": "n각형의 내각의 크기의 합은 180°×n이다.",
        "example": "사각형 → 180×4 = 720°",
        "explanation": "n각형은 한 꼭짓점에서 대각선을 그으면 삼각형 n-2개로 나뉘어. 그래서 내각의 합은 180°×(n-2)야. "
                       "사각형이면 삼각형 2개로 나뉘니까 180×2=360°이고, 720°가 아니야. "
                       "n개가 아니라 n-2개의 삼각형이라는 걸 기억하자.",
        "problems": [5, 6, 7, 8, 9, 10],
        "statement": lambda n: f"{n}각형의 내각의 크기의 합을 구하시오.",
        "answer": lambda n: sp.Integer(180 * (n - 2)),
        "bug": lambda n: sp.Integer(180 * n),
    },
    "pythagoras_no_sqrt": {
        "name": "피타고라스에서 제곱근을 구하지 않음",
        "kind": "value",
        "concept": "pythagoras",
        "diagnosis": "a^2+b^2까지 구하고 제곱근을 구하지 않음 (빗변의 제곱을 빗변으로 씀)",
        "belief": "직각삼각형에서 빗변의 길이는 직각을 낀 두 변의 제곱의 합 a^2+b^2이다.",
        "example": "직각을 낀 두 변이 1cm, 2cm → 1+4=5 → 빗변 5cm",
        "explanation": "피타고라스 정리는 c^2=a^2+b^2이야. 그러니까 a^2+b^2은 빗변의 '제곱'이고, 빗변은 그 제곱근이야. "
                       "두 변이 1cm, 2cm면 c^2=5이니까 빗변은 sqrt(5)cm야. "
                       "길이를 구할 때는 마지막에 제곱근을 꼭 씌우자.",
        "problems": [[3, 4], [5, 12], [9, 12], [8, 15], [7, 24], [12, 16]],
        "statement": lambda ab: f"직각삼각형에서 직각을 낀 두 변의 길이가 {ab[0]}cm, {ab[1]}cm일 때, 빗변의 길이를 구하시오.",
        "answer": lambda ab: sp.sqrt(ab[0] ** 2 + ab[1] ** 2),
        "bug": lambda ab: sp.Integer(ab[0] ** 2 + ab[1] ** 2),
    },
    "hypotenuse_confusion": {
        "name": "빗변을 직각을 낀 변으로 착각",
        "kind": "value",
        "concept": "pythagoras",
        "diagnosis": "빗변이 주어졌는데도 두 변의 제곱의 합을 사용 (빗변을 직각을 낀 변으로 착각)",
        "belief": "직각삼각형에서 모르는 변의 길이는 항상 아는 두 변의 제곱을 더해서 제곱근을 구한다.",
        "example": "빗변 5cm, 다른 한 변 3cm → sqrt(25+9)=sqrt(34)cm",
        "explanation": "c^2=a^2+b^2에서 c는 빗변, 즉 가장 긴 변이야. 빗변이 주어졌으면 c 자리에 넣어야 해. "
                       "빗변 5cm, 한 변 3cm라면 5^2=3^2+b^2이니까 b^2=25-9=16, b=4cm야. "
                       "sqrt(34)는 5보다 길어서 빗변보다 긴 변이 생기니까 말이 안 되지.",
        "problems": [[13, 5], [10, 6], [17, 8], [25, 7], [15, 9], [20, 12]],
        "statement": lambda ca: f"직각삼각형에서 빗변의 길이가 {ca[0]}cm, 다른 한 변의 길이가 {ca[1]}cm일 때, "
                                f"나머지 한 변의 길이를 구하시오.",
        "answer": lambda ca: sp.sqrt(ca[0] ** 2 - ca[1] ** 2),
        "bug": lambda ca: sp.sqrt(ca[0] ** 2 + ca[1] ** 2),
    },
    "area_ratio_linear": {
        "name": "넓이의 비를 닮음비 그대로 계산",
        "kind": "value",
        "concept": "similarity",
        "diagnosis": "넓이의 비를 닮음비의 제곱이 아니라 닮음비 그대로 계산",
        "belief": "닮은 도형의 넓이의 비는 닮음비와 같다.",
        "example": "닮음비 1:2, 작은 도형의 넓이 3cm^2 → 큰 도형의 넓이 6cm^2",
        "explanation": "닮음비가 1:2면 가로도 2배, 세로도 2배가 되니까 넓이는 2×2=4배가 돼. 그래서 넓이의 비는 1^2:2^2=1:4야. "
                       "작은 도형이 3cm^2면 큰 도형은 12cm^2이고, 6cm^2가 아니야. "
                       "넓이의 비는 닮음비의 제곱이라는 걸 기억하자.",
        "problems": [[2, 3, 8], [1, 2, 5], [3, 4, 18], [2, 5, 12], [1, 3, 7], [3, 5, 27]],
        "statement": lambda r: f"닮음비가 {r[0]}:{r[1]}인 두 닮은 도형에서 작은 도형의 넓이가 {r[2]}cm^2일 때, "
                               f"큰 도형의 넓이를 구하시오.",
        "answer": lambda r: sp.Rational(r[2] * r[1] ** 2, r[0] ** 2),
        "bug": lambda r: sp.Rational(r[2] * r[1], r[0]),
    },
}

UNITS = {
    "algebra": ["factor_sign", "root_sign", "divide_by_x", "positive_sqrt_only", "nonzero_product"],
    "statistics": ["median_unsorted", "mode_frequency", "variance_no_divide", "std_no_sqrt"],
    "geometry": ["polygon_angle_n", "pythagoras_no_sqrt", "hypotenuse_confusion", "area_ratio_linear"],
}


def build_problem(mid: str, index: int) -> dict:
    """오개념 mid의 index번째 시험 문제를 채점기(check_step)가 쓰는 문제 형식으로 만든다."""
    m = MISCONCEPTIONS[mid]
    text = m["problems"][index]
    if m["kind"] == "value":
        check = {"type": "value", "expected": str(m["answer"](text)),
                 "misconceptions": [{"answer": str(m["bug"](text)), "diagnosis": m["diagnosis"]}]}
        return {"id": f"{mid}-{index}", "type": "value", "statement": m["statement"](text),
                "steps": [{"goal": "값 구하기", "concept": m["concept"], "check": check}]}
    if m["kind"] == "factor":
        expr = parse_math(text)
        statement = f"다음 식을 인수분해하시오: {text}"
        check = {"type": "factor", "expected": str(sp.factor(expr)),
                 "misconceptions": [{"answer": m["bug"](expr), "diagnosis": m["diagnosis"]}]}
        return {"id": f"{mid}-{index}", "type": "factor", "statement": statement, "expression": text,
                "steps": [{"goal": "인수분해하기", "concept": m["concept"], "check": check}]}
    left, right = text.split("=")
    check = {"type": "roots", "expected": [str(r) for r in _roots(left, right)],
             "misconceptions": [{"answer": ", ".join(str(r) for r in m["bug"](left, right)),
                                 "diagnosis": m["diagnosis"]}]}
    return {"id": f"{mid}-{index}", "type": "solve", "statement": f"다음 이차방정식을 푸시오: {text}",
            "steps": [{"goal": "해 구하기", "concept": m["concept"], "check": check}]}


def correct_answer(problem: dict) -> str:
    check = problem["steps"][0]["check"]
    return ", ".join(check["expected"]) if check["type"] == "roots" else check["expected"]


def classify(problem: dict, answer: str) -> str:
    """학생 답을 correct(정답) / bug(심어 둔 오개념) / other(다른 오답) / unparseable 중 하나로 분류한다."""
    check = problem["steps"][0]["check"]
    result = check_step(problem, check, answer)
    if result.correct:
        return "correct"
    if result.diagnosis == check["misconceptions"][0]["diagnosis"]:
        return "bug"
    if result.error is ErrorType.UNPARSEABLE:
        return "unparseable"
    return "other"


def problem_message(problem: dict) -> str:
    kind = problem["steps"][0]["check"]["type"]
    return f"문제: {problem['statement']}\n(answer 형식: {ANSWER_FORMATS[kind]})"


STUDENT_INSTRUCTIONS = """너는 중학교 3학년 학생 역할을 한다. 수학 실력은 보통이다.
너는 다음과 같이 알고 있다.
- {belief}
- 예: {example}

규칙
- 위 방법이 맞다고 믿는다. 관련 문제를 풀 때는 이 방법을 그대로 쓴다.
  네가 실제로 아는 더 정확한 수학 지식은 이 학생이 모르는 것이므로 쓰지 않는다.
- 상대의 설명을 듣고 납득되면 그때부터 생각을 바꿀 수 있다. 납득되지 않으면 원래 방법을 유지한다.
- message에는 학생 말투로 풀이 과정이나 하고 싶은 말을 1~3문장 쓴다.
- answer에는 최종 답만 지정된 형식으로 쓴다. 거듭제곱은 ^, 제곱근은 sqrt()로 쓴다. 답할 문제가 없으면 빈 문자열로 둔다."""

STUDENT_SCHEMA = {
    "type": "object",
    "properties": {"message": {"type": "string"}, "answer": {"type": "string"}},
    "required": ["message", "answer"],
    "additionalProperties": False,
}


def student_completer(mid: str, model: str, effort: str | None, usage: llm.Usage | None = None) -> llm.JsonCompleter:
    m = MISCONCEPTIONS[mid]
    instructions = STUDENT_INSTRUCTIONS.format(belief=m["belief"], example=m["example"])
    return llm.json_completer(model, instructions, STUDENT_SCHEMA, "student", effort, usage)


def as_history(reply: dict) -> dict:
    """학생 응답을 다음 호출의 대화 기록(assistant 메시지)으로 바꾼다."""
    text = reply["message"] + (f"\n답: {reply['answer']}" if reply["answer"] else "")
    return {"role": "assistant", "content": text}
