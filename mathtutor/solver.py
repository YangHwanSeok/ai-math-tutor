import json
import os
from dataclasses import dataclass, field

from mathtutor import llm
from mathtutor.checker import CheckResult, check_step

ANSWER_FORMATS = {
    "factor": "인수분해된 식만 쓴다. 예: (x-2)(x-3), 3x(x-2), (x+3)^2",
    "roots": "해를 쉼표로 구분해 쓴다. 중근은 한 번만 쓴다. 예: 2, 3 / 1+sqrt(2), 1-sqrt(2)",
    "value": "수 하나만 쓴다. 예: 8",
}

INSTRUCTIONS = """너는 중학교 수학 문제를 단계별로 푸는 풀이기다.
- 주어진 단계 순서와 개수를 그대로 지켜서 푼다.
- work에는 그 단계의 풀이 과정을 한두 문장으로 쓴다.
- answer에는 단계별로 지정된 형식의 결과만 쓴다. 거듭제곱은 ^, 제곱근은 sqrt()로 쓴다."""

SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"work": {"type": "string"}, "answer": {"type": "string"}},
                "required": ["work", "answer"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}


@dataclass
class Attempt:
    steps: list[dict]
    results: list[CheckResult]

    @property
    def correct(self) -> bool:
        return bool(self.results) and all(r.correct for r in self.results)


@dataclass
class SolveRecord:
    problem_id: str
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def first_try_correct(self) -> bool:
        return self.attempts[0].correct

    @property
    def verified(self) -> bool:
        return self.attempts[-1].correct


def llm_solver(model: str | None = None, effort: str | None = "low", usage: llm.Usage | None = None) -> llm.JsonCompleter:
    return llm.json_completer(model or os.environ["SOLVER_MODEL"], INSTRUCTIONS, SCHEMA, "solution", effort, usage)


def build_prompt(problem: dict) -> str:
    lines = [f"문제: {problem['statement']}", "아래 단계 순서대로 풀어라."]
    for i, step in enumerate(problem["steps"], 1):
        lines.append(f"{i}. {step['goal']} (answer 형식: {ANSWER_FORMATS[step['check']['type']]})")
    return "\n".join(lines)


def verify(problem: dict, steps: list[dict]) -> list[CheckResult]:
    if len(steps) != len(problem["steps"]):
        return []
    return [check_step(problem["expression"], spec["check"], out["answer"])
            for spec, out in zip(problem["steps"], steps)]


def _feedback(problem: dict, attempt: Attempt) -> str:
    if not attempt.results:
        return f"단계 수가 {len(problem['steps'])}개여야 한다. 지정된 단계대로 다시 풀어라."
    lines = [f"{i}단계 답 '{out['answer']}'이(가) 검증에 실패했다: {r.error.value}. {r.detail}"
             for i, (out, r) in enumerate(zip(attempt.steps, attempt.results), 1) if not r.correct]
    lines.append("실패한 단계를 다시 확인하고 전체 풀이를 다시 제출하라.")
    return "\n".join(lines)


def solve_with_verification(problem: dict, complete: llm.JsonCompleter, max_attempts: int = 3) -> SolveRecord:
    """풀이를 생성하고 SymPy로 검증한다. 실패하면 실패 내용을 알려주고 최대 max_attempts번까지 다시 생성한다."""
    messages = [{"role": "user", "content": build_prompt(problem)}]
    record = SolveRecord(problem["id"])
    for _ in range(max_attempts):
        steps = complete(messages)["steps"]
        attempt = Attempt(steps, verify(problem, steps))
        record.attempts.append(attempt)
        if attempt.correct:
            break
        messages += [
            {"role": "assistant", "content": json.dumps({"steps": steps}, ensure_ascii=False)},
            {"role": "user", "content": _feedback(problem, attempt)},
        ]
    return record
