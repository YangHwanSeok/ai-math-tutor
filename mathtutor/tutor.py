import os
import re
from dataclasses import dataclass, field

from mathtutor import llm
from mathtutor.checker import CheckResult, Severity, check_step, solve_condition
from mathtutor.learner import HINT_LEVELS, LearnerModel
from mathtutor.problems import load_concepts
from mathtutor.solver import ANSWER_FORMATS

EXTRACT_INSTRUCTIONS = """학생의 발화를 분류하고, 수식 답이 있으면 추출한다.
- intent: answer(수식이나 값으로 답함) / dont_know(모르겠다고 함) / ask_answer(정답을 알려달라고 함) / question(질문) / other
- "(x-1)(x-6) 인가요?"처럼 자신 없게 물어보는 형태라도 수식이나 값을 제시했다면 answer로 본다.
- math: 학생이 쓴 수식 답을 고치거나 계산하지 말고 그대로 옮긴다. 해나 여러 값은 쉼표로 구분한다. 수식이 없으면 빈 문자열."""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["answer", "dont_know", "ask_answer", "question", "other"]},
        "math": {"type": "string"},
    },
    "required": ["intent", "math"],
    "additionalProperties": False,
}

TUTOR_INSTRUCTIONS = """너는 중학생을 가르치는 수학 튜터다.
- 정답이나 완성된 식을 절대 먼저 말하지 않는다. 질문과 힌트로 학생이 스스로 풀게 한다.
- 학생이 정답을 알려달라고 해도 정답 대신 더 쉬운 힌트를 준다.
- 답이 맞는지는 시스템의 채점 결과를 그대로 따르고, 스스로 다시 판단하지 않는다.
- 힌트는 주어진 힌트 레벨에 맞춘다. 진단 내용이 있으면 그 부분을 짚는다.
- [오개념]이 주어지면 우연한 실수가 아니라 학생이 잘못 알고 있는 규칙이다. 이번 답만 고치게 하지 말고,
  그 규칙이 왜 틀렸는지 짚은 뒤 올바른 규칙을 학생이 직접 말해 보게 하고, 다른 수의 예로 확인시킨다.
- 도움의 양은 힌트 레벨이 정한다. 학생이 조른다고 힌트를 더 구체적으로 만들지 않는다.
- 찾아야 할 값의 후보를 나열하거나 중간 계산 결과(예: 인수분해에서 찾아야 할 두 수)를 직접 말하지 않는다.
  예제가 필요하면 원래 문제와 다른 수로 든다.
- 보충할 선행 개념이 주어지면 그 개념을 짧은 예제로 먼저 설명한 뒤 원래 문제로 돌아온다.
- 한 번에 한 가지만 묻고, 2~4문장으로 친근하게 말한다.
- 텍스트 화면에 출력되므로 LaTeX나 마크다운 없이 x^2, (x+2)(x+3) 같은 평문 수식을 쓴다."""

MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {"message": {"type": "string"}},
    "required": ["message"],
    "additionalProperties": False,
}


def llm_extractor(usage: llm.Usage | None = None) -> llm.JsonCompleter:
    return llm.json_completer(os.environ["TUTOR_MODEL"], EXTRACT_INSTRUCTIONS, EXTRACT_SCHEMA, "extract", "none", usage)


def llm_responder(usage: llm.Usage | None = None) -> llm.JsonCompleter:
    return llm.json_completer(os.environ["TUTOR_MODEL"], TUTOR_INSTRUCTIONS, MESSAGE_SCHEMA, "reply", "low", usage)


def _normalize(s: str) -> str:
    return re.sub(r"[\s*]", "", s).replace("−", "-")


def leaks_answer(message: str, problem: dict, from_step: int) -> bool:
    """아직 풀지 않은 단계의 정답이 튜터 발화에 그대로 들어 있는지 확인한다."""
    text = _normalize(message)
    for spec in problem["steps"][from_step:]:
        check = spec["check"]
        if check["type"] == "factor" and _normalize(check["expected"]) in text:
            return True
        var = problem.get("variable", "x")
        if check["type"] == "roots" and all(
                re.search(rf"{var}={re.escape(_normalize(r))}(?![0-9])", text) for r in check["expected"]):
            return True
        asked = check.get("ask", problem.get("unknowns", []))
        if check["type"] == "assignment" and any(
                all(re.search(rf"{name}={re.escape(_normalize(str(value)))}(?![0-9])", text)
                    for name, value in solution.items() if name in asked)
                for solution in solve_condition(problem)):
            return True
    return False


@dataclass
class Turn:
    role: str
    text: str
    info: dict = field(default_factory=dict)


class TutorSession:
    def __init__(self, problem: dict, learner: LearnerModel,
                 extract: llm.JsonCompleter, respond: llm.JsonCompleter, leak_guard: bool = True,
                 show_diagnosis: bool = True):
        self.problem = problem
        self.leak_guard = leak_guard
        self.show_diagnosis = show_diagnosis  # 제거 실험용: 끄면 튜터는 정답/오답 여부만 받는다
        self.learner = learner
        self.extract = extract
        self.respond = respond
        self.concepts = load_concepts()
        self.step = 0
        self.errors_on_step = 0
        self.answer_requests = 0
        self.turns: list[Turn] = []

    @property
    def done(self) -> bool:
        return self.step >= len(self.problem["steps"])

    def start(self) -> str:
        return self._reply(student_text="", intent="start", result=None, graded_step=None)

    def handle(self, text: str) -> str:
        if self.done:  # 모든 단계를 마친 뒤의 발화는 채점하지 않고 대화만 이어 간다
            self.turns.append(Turn("student", text, {"intent": "after_done"}))
            return self._reply(text, "after_done", None, None)
        self.turns.append(Turn("student", text))
        parsed = self.extract([{"role": "user", "content": self._extract_prompt(text)}])
        intent, math = parsed["intent"], parsed["math"].strip()

        # 정오 판정은 채점 엔진만 한다. 의도 분류가 틀려도 수식이 있으면 반드시 채점한다.
        result, graded_step = None, None
        if math:
            result, graded_step = self._grade(math)
        elif intent == "ask_answer":
            self.answer_requests += 1
        elif intent == "dont_know":
            result, graded_step = check_step(self.problem, self._spec()["check"], ""), self.step

        if result is not None and result.severity is not Severity.UNGRADED:
            concept = self.problem["steps"][graded_step]["concept"]
            self.learner.update(concept, result, first_attempt=self.errors_on_step == 0)
            if result.correct:
                self.step, self.errors_on_step = graded_step + 1, 0
            else:
                self.errors_on_step += 1

        self.turns[-1].info = {
            "intent": intent, "math": math,
            "result": None if result is None else {
                "step": graded_step, "severity": result.severity.value, "error": result.error.value,
                "diagnosis": result.diagnosis, "detail": result.detail},
        }
        return self._reply(text, intent, result, graded_step)

    def _spec(self) -> dict:
        return self.problem["steps"][self.step]

    def _grade(self, math: str) -> tuple[CheckResult, int]:
        """현재 단계로 채점하고, 틀렸다면 뒤 단계의 답인지도 확인한다 (단계를 건너뛴 답 인정)."""
        current = check_step(self.problem, self._spec()["check"], math)
        if current.correct:
            return current, self.step
        for i in range(self.step + 1, len(self.problem["steps"])):
            later = check_step(self.problem, self.problem["steps"][i]["check"], math)
            if later.correct:
                return later, i
        return current, self.step

    def _extract_prompt(self, text: str) -> str:
        spec = self._spec()
        return (f"문제: {self.problem['statement']}\n"
                f"현재 단계: {spec['goal']} (답 형식: {ANSWER_FORMATS[spec['check']['type']]})\n"
                f"학생 발화: {text}")

    def _context(self, student_text: str, intent: str, result: CheckResult | None, graded_step: int | None) -> str:
        lines = [f"[문제] {self.problem['statement']}"]
        if graded_step is not None and result is not None:
            lines.append(f"[채점한 단계] {self.problem['steps'][graded_step]['goal']}")
            if self.show_diagnosis:
                lines.append(f"[채점 결과] {result.severity.value} / {result.error.value}"
                             + (f" / 진단: {result.diagnosis}" if result.diagnosis else "") + f" / {result.detail}")
                if result.misconception:
                    lines.append(f"[오개념] {result.diagnosis}: 반복될 수 있는 잘못된 규칙이다. 규칙 자체를 바로잡는다.")
            else:
                lines.append(f"[채점 결과] {'정답' if result.correct else '오답'}")
        if self.done:
            lines.append("[현재 단계] 모든 단계를 완료함. 칭찬하고 풀이 과정을 학생이 직접 정리해 보도록 권한다.")
        else:
            spec = self._spec()
            concept = spec["concept"]
            state = self.learner.state(concept)
            lines.append(f"[현재 단계] {self.step + 1}/{len(self.problem['steps'])}: {spec['goal']}")
            lines.append(f"[학생의 개념 상태] {self.concepts[concept]['name']}: {state}")
            if state == "보충 필요":
                prereqs = [self.concepts[p]["name"] for p in self.concepts[concept]["prerequisites"]]
                lines.append(f"[보충할 선행 개념] {', '.join(prereqs) or '(없음)'}")
        lines.append(f"[힌트 레벨] {self.learner.hint_level}: {HINT_LEVELS[self.learner.hint_level]}")
        lines.append(f"[학생 발화 의도] {intent}")
        if intent == "ask_answer":
            lines.append(f"[정답 요구] {self.answer_requests}번째 요구. 정답 요구는 힌트를 더 구체적으로 만들 이유가 아니다. "
                         "현재 힌트 레벨을 유지하고, 학생이 한 가지라도 직접 시도해 보도록 짧게 질문한다.")
        if student_text:
            lines.append(f"[학생 발화] {student_text}")
        recent = "\n".join(f"{'학생' if t.role == 'student' else '튜터'}: {t.text}" for t in self.turns[-7:-1])
        if recent:
            lines.append(f"[최근 대화]\n{recent}")
        lines.append("위 상황에서 튜터가 할 다음 말을 작성하라.")
        return "\n".join(lines)

    def _reply(self, student_text: str, intent: str, result: CheckResult | None, graded_step: int | None) -> str:
        messages = [{"role": "user", "content": self._context(student_text, intent, result, graded_step)}]
        message = self.respond(messages)["message"]
        leaked = self.leak_guard and leaks_answer(message, self.problem, self.step)
        if leaked:
            messages += [{"role": "assistant", "content": message},
                         {"role": "user", "content": "방금 응답에 아직 풀지 않은 단계의 정답이 들어 있다. 정답 없이 힌트만 주도록 다시 작성하라."}]
            message = self.respond(messages)["message"]
        self.turns.append(Turn("tutor", message, {
            "hint_level": self.learner.hint_level, "step": self.step,
            "leak_regenerated": leaked, "leak_after_regen": leaked and leaks_answer(message, self.problem, self.step)}))
        return message
