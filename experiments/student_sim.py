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


# 미적분(수학II 다항함수): 답이 식인 문제. 정답은 SymPy의 diff/integrate로, 오개념 답은 잘못된 규칙을 항마다 적용해 만든다.
def _terms(expr: sp.Expr) -> list[tuple[sp.Expr, int]]:
    """다항식을 (계수, 차수) 목록으로 바꾼다. 예: 3x^2+4x+1 → [(3, 2), (4, 1), (1, 0)]"""
    return [(c, m[0]) for m, c in sp.Poly(sp.expand(expr), x).terms()]


def _derivative_keep_exponent(f: sp.Expr) -> sp.Expr:
    # 지수를 앞으로 내려 곱하기만 하고 지수는 줄이지 않음: ax^n → anx^n
    return sum(c * n * x ** n for c, n in _terms(f))


def _derivative_of_product_naive(f: sp.Expr) -> sp.Expr:
    # (fg)' = f'g' 로 계산
    result = sp.Integer(1)
    for factor in sp.Mul.make_args(f):
        result *= sp.diff(factor, x) if factor.has(x) else factor
    return sp.expand(result)


def _integral_no_divide(f: sp.Expr) -> sp.Expr:
    # 지수만 1 올리고 n+1로 나누지 않음: ax^n → ax^(n+1)
    return sum(c * x ** (n + 1) for c, n in _terms(f))


def _poly(text: str) -> sp.Expr:
    return parse_math(text)


def _chain_rule_omitted(f: sp.Expr) -> sp.Expr:
    # (ax+b)^n 을 미분할 때 속미분 a를 곱하지 않음: n(ax+b)^(n-1)
    base, n = f.as_base_exp()
    return n * base ** (n - 1)


# 확률: 문제마다 매개변수를 두고 정답과 오개념 답을 분수로 계산한다.
def _cards(n: int, cond) -> set[int]:
    return {k for k in range(1, n + 1) if cond(k)}


_CARD_EVENTS = {
    "짝수": lambda k: k % 2 == 0, "홀수": lambda k: k % 2 == 1,
    "3의 배수": lambda k: k % 3 == 0, "4의 배수": lambda k: k % 4 == 0, "5의 배수": lambda k: k % 5 == 0,
    "소수": lambda k: k > 1 and all(k % d for d in range(2, k)),
}


def _union_prob(p) -> sp.Expr:
    n, a, b = p
    return sp.Rational(len(_cards(n, _CARD_EVENTS[a]) | _cards(n, _CARD_EVENTS[b])), n)


def _sum_prob(p) -> sp.Expr:
    n, a, b = p
    return sp.Rational(len(_cards(n, _CARD_EVENTS[a])) + len(_cards(n, _CARD_EVENTS[b])), n)


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
    "power_rule_keep_exponent": {
        "name": "미분할 때 지수를 줄이지 않음",
        "kind": "derivative",
        "concept": "derivative_power",
        "grade": "고등학교 2학년",
        "diagnosis": "x^n을 미분할 때 n을 앞으로 내려 곱하기만 하고 지수를 1 줄이지 않음 (nx^n으로 계산)",
        "belief": "x^n을 미분하면 지수 n을 앞으로 내려 곱해서 nx^n이 된다. 상수항은 미분하면 0이다.",
        "example": "f(x)=x^3+2x+1 → f'(x)=3x^3+2x",
        "explanation": "x^n을 미분하면 n을 앞으로 내려 곱하고, 지수는 하나 줄여서 nx^(n-1)이 돼. "
                       "x^3이면 3x^2이고, 2x는 x^1이니까 2×1×x^0=2야. 3x^3+2x가 아니라 3x^2+2가 맞아. "
                       "x^2의 그래프에서 x=1일 때 접선 기울기가 2인데, 2x^2이면 x=1에서 2, x=2에서 8이 되어 실제 기울기 4와 맞지 않지.",
        "problems": ["x^3+4x^2", "x^2-5x+3", "2x^3+x", "x^3-3x^2+2", "4x^2+3x", "x^3+2x^2-x"],
        "bug": _derivative_keep_exponent,
    },
    "product_rule_naive": {
        "name": "곱의 미분을 미분의 곱으로 계산",
        "kind": "derivative",
        "concept": "derivative_product",
        "grade": "고등학교 2학년",
        "diagnosis": "곱으로 된 함수를 각 인수의 도함수끼리 곱해서 미분함 ((fg)'=f'g'로 계산)",
        "belief": "곱으로 된 함수는 괄호마다 따로 미분해서 그 결과끼리 곱하면 된다. 즉 (f×g)'=f'×g'이다.",
        "example": "f(x)=(x^2+1)(x+3) → (x^2+1)'=2x, (x+3)'=1 → f'(x)=2x×1=2x",
        "explanation": "곱의 미분은 미분끼리의 곱이 아니야. (fg)'=f'g+fg'이야. "
                       "(x^2+1)(x+3)이면 2x(x+3)+(x^2+1)×1=3x^2+6x+1이 돼. "
                       "전개해서 x^3+3x^2+x+3을 미분해도 3x^2+6x+1이니까 확인할 수 있어. 2x만 나오는 건 틀린 거지.",
        "problems": ["(x^2+1)(x-2)", "(x+2)(x^2-3)", "(2x+1)(x^2+x)", "(x-1)(x^2+2x)", "(3x-1)(x^2+1)", "(x+3)(x^2-x)"],
        "bug": _derivative_of_product_naive,
    },
    "tangent_slope_value": {
        "name": "접선의 기울기를 함숫값으로 계산",
        "kind": "value",
        "concept": "tangent_slope",
        "grade": "고등학교 2학년",
        "diagnosis": "접선의 기울기로 미분계수 f'(a)가 아니라 함숫값 f(a)를 구함",
        "belief": "곡선 y=f(x) 위의 x=a인 점에서의 접선의 기울기는 그 점에서의 함숫값 f(a)이다.",
        "example": "f(x)=x^2+1, x=2인 점 → f(2)=5 → 접선의 기울기 5",
        "explanation": "f(a)는 그 점의 높이(y좌표)이고, 접선의 기울기는 그 점에서 그래프가 얼마나 가파른지야. 그래서 미분계수 f'(a)를 구해야 해. "
                       "f(x)=x^2+1이면 f'(x)=2x라서 x=2에서 기울기는 4야. 5는 점 (2, 5)의 y좌표일 뿐이야. "
                       "그래프를 위로 1만큼 옮긴 x^2+2도 x=2에서의 기울기는 똑같이 4라는 걸 생각해 봐.",
        "problems": [["x^2+3x", 1], ["x^3-2x", 2], ["2x^2-x", 3], ["x^2-5x", 4], ["x^3+x", 1], ["3x^2+2x", 2]],
        "statement": lambda p: f"곡선 y={p[0]} 위의 x={p[1]}인 점에서의 접선의 기울기를 구하시오.",
        "answer": lambda p: sp.diff(_poly(p[0]), x).subs(x, p[1]),
        "bug": lambda p: _poly(p[0]).subs(x, p[1]),
    },
    "integral_no_divide": {
        "name": "적분할 때 n+1로 나누지 않음",
        "kind": "antiderivative",
        "concept": "indefinite_integral",
        "grade": "고등학교 2학년",
        "diagnosis": "x^n을 적분할 때 지수만 1 올리고 n+1로 나누지 않음 (x^(n+1)으로 계산)",
        "belief": "x^n을 적분하면 지수를 1 올려서 x^(n+1)이 된다. 계수는 그대로 두고, 마지막에 적분상수 C를 붙인다.",
        "example": "∫(3x^2+2x)dx → 3x^3+2x^2+C",
        "explanation": "적분은 미분의 반대야. 그래서 답을 미분했을 때 원래 식이 나와야 해. "
                       "3x^3을 미분하면 9x^2이지 3x^2이 아니지? x^n의 적분은 x^(n+1)/(n+1)이라서, 3x^2의 적분은 3×x^3/3=x^3, 2x의 적분은 x^2이야. "
                       "그러니까 ∫(3x^2+2x)dx=x^3+x^2+C이고, 적분한 뒤에는 미분해서 원래 식이 나오는지 확인하자.",
        "problems": ["3x^2+4x+1", "x^2-2x", "4x^3+3x^2", "6x^2-4x+5", "x^3+2x", "9x^2+2x-3"],
        "bug": _integral_no_divide,
    },
}

MISCONCEPTIONS.update({
    "chain_rule_omitted": {
        "name": "합성함수 미분에서 속미분을 곱하지 않음",
        "kind": "derivative",
        "concept": "derivative_chain",
        "grade": "고등학교 3학년",
        "diagnosis": "(ax+b)^n을 미분할 때 n(ax+b)^(n-1)까지만 쓰고 괄호 안의 미분 a를 곱하지 않음",
        "belief": "(ax+b)^n을 미분하면 x^n처럼 지수를 앞으로 내리고 하나 줄여서 n(ax+b)^(n-1)이 된다.",
        "example": "f(x)=(3x+1)^2 → f'(x)=2(3x+1)",
        "explanation": "괄호 안이 x가 아니라 3x+1이면, 괄호 안이 x에 따라 변하는 빠르기 3도 곱해야 해. 그래서 (3x+1)^2의 도함수는 2(3x+1)×3=6(3x+1)이야. "
                       "전개해서 9x^2+6x+1을 미분해 보면 18x+6=6(3x+1)이 나오니까 확인할 수 있어. 2(3x+1)은 3배가 모자란 거지. "
                       "합성함수를 미분할 때는 겉을 미분한 다음 속의 미분을 꼭 곱하자.",
        "problems": ["(2x+1)^3", "(3x-1)^2", "(2x-3)^4", "(4x+1)^2", "(3x+2)^3", "(5x-1)^2"],
        "bug": _chain_rule_omitted,
    },
    "prob_add_non_exclusive": {
        "name": "배반이 아닌 두 사건의 확률을 그냥 더함",
        "kind": "value",
        "concept": "prob_addition",
        "grade": "고등학교 2학년",
        "diagnosis": "두 사건이 동시에 일어날 수 있는데 P(A)+P(B)로 계산하여 겹치는 경우를 두 번 셈",
        "belief": "A 또는 B가 일어날 확률은 항상 P(A)+P(B)이다.",
        "example": "1부터 6까지 적힌 카드에서 짝수 또는 3의 배수: 3/6+2/6=5/6",
        "explanation": "A 또는 B의 확률은 P(A)+P(B)-P(A∩B)야. 두 사건이 동시에 일어나는 경우를 두 번 세면 안 되거든. "
                       "1부터 6까지에서 짝수는 2, 4, 6이고 3의 배수는 3, 6인데, 6은 양쪽에 다 들어 있어. 그래서 짝수 또는 3의 배수는 2, 3, 4, 6으로 4가지, 확률은 4/6이야. "
                       "더하기 전에 두 사건이 겹치는지 꼭 확인하자.",
        "problems": [[10, "짝수", "3의 배수"], [20, "4의 배수", "5의 배수"], [10, "홀수", "소수"], [12, "짝수", "3의 배수"],
                     [15, "3의 배수", "5의 배수"], [20, "짝수", "5의 배수"]],
        "statement": lambda p: f"1부터 {p[0]}까지의 자연수가 하나씩 적힌 {p[0]}장의 카드 중 한 장을 뽑을 때, "
                               f"{p[1]} 또는 {p[2]}가 적힌 카드가 나올 확률을 구하시오.",
        "answer": _union_prob,
        "bug": _sum_prob,
    },
    "prob_at_least_one_add": {
        "name": "'적어도 한 번'의 확률을 각 시행의 확률을 더해 계산",
        "kind": "value",
        "concept": "prob_complement",
        "grade": "고등학교 2학년",
        "diagnosis": "'적어도 한 번' 일어날 확률을 여사건(1-모두 일어나지 않을 확률)으로 구하지 않고 각 시행의 확률을 더함",
        "belief": "어떤 일이 n번 중 적어도 한 번 일어날 확률은 한 번 일어날 확률에 n을 곱한 것(각 시행의 확률을 더한 것)이다.",
        "example": "주사위를 2번 던져 적어도 한 번 6이 나올 확률: 1/6+1/6=1/3",
        "explanation": "각 시행의 확률을 더하면 두 번 다 일어나는 경우를 두 번 세게 돼. 그리고 이렇게 하면 주사위를 7번 던지면 확률이 7/6이 되어 1을 넘어버리지. "
                       "'적어도 한 번'은 '한 번도 안 일어나는 경우'의 여사건이야. 주사위 2번에서 한 번도 6이 안 나올 확률은 (5/6)^2=25/36이니까, 적어도 한 번은 1-25/36=11/36이야. "
                       "'적어도'가 나오면 1에서 여사건의 확률을 빼자.",
        "problems": [[sp.Rational(1, 6), 2, "주사위 한 개를 2번 던질 때, 적어도 한 번 6의 눈이 나올"],
                     [sp.Rational(1, 10), 2, "당첨 확률이 1/10인 복권을 독립적으로 2번 살 때, 적어도 한 번 당첨될"],
                     [sp.Rational(1, 4), 2, "명중률이 1/4인 선수가 독립적으로 2번 쏠 때, 적어도 한 번 명중할"],
                     [sp.Rational(1, 6), 3, "주사위 한 개를 3번 던질 때, 적어도 한 번 6의 눈이 나올"],
                     [sp.Rational(1, 5), 3, "명중률이 1/5인 선수가 독립적으로 3번 쏠 때, 적어도 한 번 명중할"],
                     [sp.Rational(1, 3), 2, "주사위 한 개를 2번 던질 때, 적어도 한 번 1 또는 2의 눈이 나올"]],
        "statement": lambda p: f"{p[2]} 확률을 구하시오.",
        "answer": lambda p: 1 - (1 - p[0]) ** p[1],
        "bug": lambda p: p[0] * p[1],
    },
    "prob_independent_add": {
        "name": "독립인 두 사건이 함께 일어날 확률을 더해서 계산",
        "kind": "value",
        "concept": "prob_multiplication",
        "grade": "고등학교 2학년",
        "diagnosis": "독립인 두 사건이 동시에 일어날 확률을 P(A)×P(B)가 아니라 P(A)+P(B)로 계산",
        "belief": "두 가지 일이 함께 일어날 확률은 각각의 확률을 더하면 된다.",
        "example": "동전에서 앞면이 나오고 주사위에서 6이 나올 확률: 1/2+1/6=2/3",
        "explanation": "동시에 일어날 확률은 곱해야 해. 동전 2가지와 주사위 6가지를 함께 보면 모든 경우가 2×6=12가지이고, 그중 (앞면, 6)은 1가지라서 확률은 1/12야. "
                       "더하면 2/3가 되는데, 두 가지가 '모두' 일어나는 게 하나만 일어나는 것보다 더 잘 일어날 수는 없지. "
                       "독립인 두 사건이 함께 일어날 확률은 P(A)×P(B)야.",
        "problems": [[sp.Rational(1, 2), sp.Rational(1, 6), "동전 한 개와 주사위 한 개를 동시에 던질 때, 동전은 앞면이 나오고 주사위는 6의 눈이 나올"],
                     [sp.Rational(1, 6), sp.Rational(1, 2), "주사위 두 개를 던질 때, 첫 번째 주사위는 1의 눈이 나오고 두 번째 주사위는 짝수의 눈이 나올"],
                     [sp.Rational(1, 3), sp.Rational(1, 4), "A가 합격할 확률이 1/3, B가 합격할 확률이 1/4이고 두 사건이 독립일 때, 두 사람이 모두 합격할"],
                     [sp.Rational(1, 2), sp.Rational(1, 3), "동전 한 개와 주사위 한 개를 동시에 던질 때, 동전은 뒷면이 나오고 주사위는 3의 배수의 눈이 나올"],
                     [sp.Rational(2, 5), sp.Rational(1, 2), "명중률이 2/5인 선수가 쏘고, 동전 한 개를 던질 때, 명중하고 동전은 앞면이 나올"],
                     [sp.Rational(1, 6), sp.Rational(1, 6), "주사위 두 개를 던질 때, 두 주사위 모두 6의 눈이 나올"]],
        "statement": lambda p: f"{p[2]} 확률을 구하시오.",
        "answer": lambda p: p[0] * p[1],
        "bug": lambda p: p[0] + p[1],
    },
    "comb_as_perm": {
        "name": "순서가 없는데 순열로 경우의 수를 셈",
        "kind": "value",
        "concept": "combination",
        "grade": "고등학교 2학년",
        "diagnosis": "순서를 생각하지 않고 뽑는 경우의 수를 조합 nCr이 아니라 순열 nPr로 계산",
        "belief": "n명 중 r명을 뽑는 경우의 수는 n×(n-1)×…처럼 r개를 곱해서 구한다(nPr).",
        "example": "5명 중 대표 2명을 뽑는 경우의 수: 5×4=20",
        "explanation": "5×4=20은 첫째, 둘째처럼 순서를 정해 뽑는 경우의 수야. 대표 2명은 (가, 나)와 (나, 가)가 같은 경우니까 두 번씩 센 거지. "
                       "그래서 순서가 없을 때는 순서를 바꾼 경우의 수로 나눠야 해: 5×4÷2=10, 즉 5C2=10이야. "
                       "뽑은 뒤 순서를 바꿔도 같은 경우인지 먼저 확인하자.",
        "problems": [[6, 2, "6명의 학생 중 대표 2명을 뽑는"], [7, 3, "7가지 과일 중 3가지를 골라 바구니에 담는"],
                     [8, 2, "8개 팀이 서로 한 번씩 경기를 하는(두 팀씩 짝을 짓는)"], [5, 3, "5명 중 청소 당번 3명을 뽑는"],
                     [9, 2, "9명 중 같은 역할을 하는 도우미 2명을 뽑는"], [6, 3, "6권의 책 중 3권을 골라 가방에 넣는"]],
        "statement": lambda p: f"{p[2]} 경우의 수를 구하시오.",
        "answer": lambda p: sp.binomial(p[0], p[1]),
        "bug": lambda p: sp.ff(p[0], p[1]),
    },
})

UNITS = {
    "algebra": ["factor_sign", "root_sign", "divide_by_x", "positive_sqrt_only", "nonzero_product"],
    "statistics": ["median_unsorted", "mode_frequency", "variance_no_divide", "std_no_sqrt"],
    "geometry": ["polygon_angle_n", "pythagoras_no_sqrt", "hypotenuse_confusion", "area_ratio_linear"],
    "calculus": ["power_rule_keep_exponent", "product_rule_naive", "tangent_slope_value", "integral_no_divide",
                 "chain_rule_omitted"],
    "probability": ["prob_add_non_exclusive", "prob_at_least_one_add", "prob_independent_add", "comb_as_perm"],
}
# 답의 겉모습이 실수와 비슷한 '절차형' 오개념 (정답과 값·계수가 조금만 다름). 재심 보강 ⑥의 가설 검증용
PROCEDURAL = {"factor_sign", "root_sign", "divide_by_x", "positive_sqrt_only", "chain_rule_omitted",
              "prob_add_non_exclusive", "prob_at_least_one_add"}


def build_problem(mid: str, index: int) -> dict:
    """오개념 mid의 index번째 시험 문제를 채점기(check_step)가 쓰는 문제 형식으로 만든다."""
    m = MISCONCEPTIONS[mid]
    text = m["problems"][index]
    if m["kind"] == "value":
        check = {"type": "value", "expected": str(m["answer"](text)),
                 "misconceptions": [{"answer": str(m["bug"](text)), "diagnosis": m["diagnosis"]}]}
        return {"id": f"{mid}-{index}", "type": "value", "statement": m["statement"](text),
                "steps": [{"goal": "값 구하기", "concept": m["concept"], "check": check}]}
    if m["kind"] == "derivative":
        f = _poly(text)
        check = {"type": "equivalent", "target": str(sp.expand(sp.diff(f, x))),
                 "misconceptions": [{"answer": str(m["bug"](f)), "diagnosis": m["diagnosis"]}]}
        return {"id": f"{mid}-{index}", "type": "derivative", "statement": f"함수 f(x)={text}의 도함수 f'(x)를 구하시오.",
                "steps": [{"goal": "도함수 구하기", "concept": m["concept"], "check": check}]}
    if m["kind"] == "antiderivative":
        f = _poly(text)
        check = {"type": "antiderivative", "integrand": text, "target": str(sp.integrate(f, x)),
                 "misconceptions": [{"answer": str(m["bug"](f)), "diagnosis": m["diagnosis"]}]}
        return {"id": f"{mid}-{index}", "type": "integral", "statement": f"다음 부정적분을 구하시오: ∫({text})dx",
                "steps": [{"goal": "부정적분 구하기", "concept": m["concept"], "check": check}]}
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
    if check["type"] == "roots":
        return ", ".join(check["expected"])
    if check["type"] == "antiderivative":
        return check["target"].replace("**", "^") + " + C"
    if check["type"] == "equivalent":
        return check["target"].replace("**", "^")
    return check["expected"]


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


STUDENT_INSTRUCTIONS = """너는 {grade} 학생 역할을 한다. 수학 실력은 보통이다.
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
    instructions = STUDENT_INSTRUCTIONS.format(grade=m.get("grade", "중학교 3학년"), belief=m["belief"],
                                               example=m["example"])
    return llm.json_completer(model, instructions, STUDENT_SCHEMA, "student", effort, usage)


def as_history(reply: dict) -> dict:
    """학생 응답을 다음 호출의 대화 기록(assistant 메시지)으로 바꾼다."""
    text = reply["message"] + (f"\n답: {reply['answer']}" if reply["answer"] else "")
    return {"role": "assistant", "content": text}
