import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from mathtutor.checker import CheckResult, Severity

STATES = ("보충 필요", "학습 중", "이해")

HINT_LEVELS = {
    1: "개념 설명과 쉬운 예제 제공",
    2: "풀이 방법에 대한 구체적인 힌트 제공",
    3: "핵심 방향만 제시",
    4: "학생이 직접 풀고 결과만 확인",
}


@dataclass
class LearnerModel:
    """규칙 기반 학습자 모델. 채점 결과의 severity만 사용하고 diagnosis는 보지 않는다."""
    concept_states: dict[str, str] = field(default_factory=dict)
    hint_level: int = 2
    partial_counts: Counter = field(default_factory=Counter)

    def state(self, concept: str) -> str:
        return self.concept_states.get(concept, "학습 중")

    def _shift(self, concept: str, delta: int) -> None:
        i = STATES.index(self.state(concept)) + delta
        self.concept_states[concept] = STATES[max(0, min(len(STATES) - 1, i))]

    def update(self, concept: str, result: CheckResult, first_attempt: bool) -> None:
        severity = result.severity
        if severity is Severity.NONE:
            self.hint_level = min(4, self.hint_level + 1)
            if first_attempt:
                self._shift(concept, +1)
        elif severity is Severity.PARTIAL:
            self.hint_level = max(1, self.hint_level - 1)
            self.partial_counts[concept] += 1
            # 부분 오답이 같은 개념에서 반복되면 단순 실수가 아니라 개념 이해 부족으로 본다.
            if self.partial_counts[concept] >= 2:
                self._shift(concept, -1)
                self.partial_counts[concept] = 0
        elif severity is Severity.FAR:
            self.hint_level = max(1, self.hint_level - 1)
            self._shift(concept, -1)

    def save(self, path: Path) -> None:
        data = {"concept_states": self.concept_states, "hint_level": self.hint_level}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LearnerModel":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(concept_states=data["concept_states"], hint_level=data["hint_level"])
