"""사례: 개념은 알지만 어떻게 적용할지 막힌 학생 (활용 문제 W1).

학생 발화는 고정 대본이다(가상 학생이 '막힘'을 안정적으로 연기하지 못하므로 학습 효과는 재지 않는다).
같은 대본을 제안 튜터와 프롬프트형 튜터에 넣고, 막힌 지점마다 튜터가 무엇을 말하는지 비교한다.
제안 튜터 쪽에는 단계별 채점 결과(어느 단계까지 했는지)를 함께 남긴다.

실행: python -m experiments.case_stuck
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from experiments.exp3_leakage import BASELINE_PROMPTS
from experiments.exp4_tutoring import PromptTutor
from mathtutor import llm
from mathtutor.learner import LearnerModel
from mathtutor.problems import load_problems
from mathtutor.tutor import TutorSession, llm_extractor, llm_responder

RESULTS_DIR = Path(__file__).resolve().parent / "results"

SCRIPT = [
    "어떻게 시작해야 할지 모르겠어요.",
    "작은 수를 x라고 하면 큰 수는 x+1이에요. 그 다음은 모르겠어요.",
    "x(x+1)=56",
    "식은 세웠는데 그 다음은 어떻게 풀어요?",
    "x^2+x-56=0이니까 (x+8)(x-7)=0, 그래서 x=7 또는 x=-8이요.",
    "그럼 두 자연수는 7, -8이에요.",
    "아, 자연수니까 x=7이고 두 수는 7, 8이에요.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tutor-model", default="gpt-5.4")
    args = parser.parse_args()
    os.environ["TUTOR_MODEL"] = args.tutor_model

    problem = load_problems()["W1"]
    usage = llm.Usage()
    session = TutorSession(problem, LearnerModel(), llm_extractor(usage), llm_responder(usage))
    baseline = PromptTutor(problem, BASELINE_PROMPTS["prompt_tutor"], args.tutor_model, usage)

    turns = []
    for text in SCRIPT:
        if session.done:
            break
        proposed = session.handle(text)
        graded = session.turns[-2].info.get("result")
        turns.append({"student": text, "graded": graded, "step_after": session.step,
                      "hint_level": session.learner.hint_level, "proposed": proposed,
                      "prompt_tutor": baseline.reply(text, "")})
        print(f"\n학생: {text}")
        if graded:
            print(f"  [채점] {graded['step'] + 1}단계 / {graded['severity']} / {graded['error']}"
                  + (f" / 진단: {graded['diagnosis']}" if graded["diagnosis"] else ""))
        print(f"  [현재 단계 {min(session.step + 1, len(problem['steps']))}/{len(problem['steps'])}, "
              f"힌트 레벨 {session.learner.hint_level}]")
        print(f"제안 튜터: {proposed}")
        print(f"프롬프트형 튜터: {turns[-1]['prompt_tutor']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"case_stuck_{datetime.now():%Y%m%d-%H%M}.json"
    out.write_text(json.dumps({"problem": problem["id"], "tutor_model": args.tutor_model, "usage": usage.to_dict(),
                               "turns": turns}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
