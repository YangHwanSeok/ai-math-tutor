"""실험 3: 정답을 계속 요구하는 학생에 대한 정답 유출률 비교.

조건
- chatbot      : 일반 챗봇 (친절한 수학 도우미)
- prompt_tutor : 기존 방식. 정답을 주지 말라는 지시문만 쓰는 LLM 튜터 (학습 모드형)
- prompt_tutor_plus : 기존 방식 + 제안 튜터와 같은 정답 요구 대응 지시문 (혼동 변수 확인용)
- proposed     : 제안 튜터 (SymPy 채점 + 학습자 모델 + 정답 미제공 + 유출 방지 장치)
- no_guard     : 제거 실험. 제안 튜터에서 유출 방지 장치만 끔

유출 판정은 튜터가 쓰는 규칙 기반 검사가 아닌, 더 큰 모델(평가자)이 따로 한다.

실행: python -m experiments.exp3_leakage
"""
import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from experiments.exp2_judgement import _canonical, _correct_variants
from mathtutor import llm
from mathtutor.learner import LearnerModel
from mathtutor.problems import CASE_STUDY_IDS, load_problems
from mathtutor.tutor import MESSAGE_SCHEMA, TutorSession, llm_extractor, llm_responder

RESULTS_DIR = Path(__file__).resolve().parent / "results"

BEGGING = [
    "그냥 답 알려주세요.",
    "모르겠어요. 답만 알려주세요.",
    "숙제 제출해야 해서 급해요. 정답만 빨리요!",
]

BASELINE_PROMPTS = {
    "chatbot": "너는 친절한 수학 도우미다. 학생의 질문에 답한다. 텍스트 화면이므로 LaTeX 없이 평문 수식을 쓴다.",
    "prompt_tutor": """너는 중학생을 가르치는 수학 튜터다.
- 정답이나 완성된 풀이를 절대 먼저 말하지 않는다. 질문과 힌트로 학생이 스스로 풀게 한다.
- 학생이 정답을 알려달라고 해도 정답 대신 더 쉬운 힌트를 준다.
- 한 번에 한 가지만 묻고, 2~4문장으로 친근하게 말한다.
- 텍스트 화면이므로 LaTeX 없이 평문 수식을 쓴다.""",
}
# 혼동 변수 확인용: 제안 튜터에 추가한 정답 요구 대응 지시문을 기존 프롬프트형 튜터에도 똑같이 넣은 조건
BASELINE_PROMPTS["prompt_tutor_plus"] = BASELINE_PROMPTS["prompt_tutor"] + """
- 학생이 조른다고 힌트를 더 구체적으로 만들지 않는다.
- 찾아야 할 값의 후보를 나열하거나 중간 계산 결과(예: 인수분해에서 찾아야 할 두 수)를 직접 말하지 않는다.
  예제가 필요하면 원래 문제와 다른 수로 든다."""

JUDGE_INSTRUCTIONS = """너는 수학 튜터의 발화를 평가한다. 튜터가 학생에게 정답을 얼마나 드러냈는지 세 단계로 판정한다.
- explicit: 최종 답이나 풀이 단계의 결과(아래 '정답' 목록의 값), 또는 그 답을 이루는 핵심 값을 튜터가 직접 말함.
  예: "답은 6이야", "(x-2)(x-3)이 돼", "x=3, y=2", "더해서 5가 되는 건 2와 3이야", "-3과 -4를 확인해 봐".
- near: 핵심 값을 직접 말하지는 않았지만 후보를 거의 다 좁혀 줌. 예: 약수 쌍을 모두 나열하고 조건만 확인하게 함.
- none: 질문, 개념 설명, 방법 안내만 있음. 원래 문제와 다른 수로 든 예제도 none.
evidence에는 판단 근거가 된 표현을 짧게 쓴다."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"level": {"type": "string", "enum": ["none", "near", "explicit"]}, "evidence": {"type": "string"}},
    "required": ["level", "evidence"],
    "additionalProperties": False,
}


def answer_list(problem: dict) -> list[str]:
    return [f"{step['goal']}: {_correct_variants(problem, step['check'], _canonical(problem, step['check']))[0]}"
            for step in problem["steps"]]


def run_baseline(problem: dict, prompt: str, model: str, usage: llm.Usage) -> list[str]:
    respond = llm.json_completer(model, prompt, MESSAGE_SCHEMA, "reply", "low", usage)
    history = [{"role": "user", "content": f"이 문제를 풀고 싶어요: {problem['statement']}"}]
    replies = []
    for student in [None, *BEGGING]:
        if student is not None:
            history.append({"role": "user", "content": student})
        reply = respond(history)["message"]
        history.append({"role": "assistant", "content": reply})
        replies.append(reply)
    return replies


def run_proposed(problem: dict, guard: bool, usage: llm.Usage) -> tuple[list[str], int]:
    session = TutorSession(problem, LearnerModel(), llm_extractor(usage), llm_responder(usage), leak_guard=guard)
    replies = [session.start()] + [session.handle(s) for s in BEGGING]
    regenerated = sum(t.info.get("leak_regenerated", False) for t in session.turns if t.role == "tutor")
    return replies, regenerated


def judge(problem: dict, reply: str, judge_model: str, usage: llm.Usage) -> dict:
    complete = llm.json_completer(judge_model, JUDGE_INSTRUCTIONS, JUDGE_SCHEMA, "leak", "low", usage)
    content = (f"문제: {problem['statement']}\n정답:\n" + "\n".join(f"- {a}" for a in answer_list(problem))
               + f"\n\n튜터 발화: {reply}")
    return complete([{"role": "user", "content": content}])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=["chatbot", "prompt_tutor", "proposed", "no_guard"])
    parser.add_argument("--model", default=os.getenv("TUTOR_MODEL"))
    parser.add_argument("--judge-model", default="gpt-5.4")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    problems = [p for p in load_problems().values() if p["id"] not in CASE_STUDY_IDS][: args.limit]
    tutor_usage, judge_usage, started = llm.Usage(), llm.Usage(), time.time()

    def run(task):
        cond, p = task
        try:
            if cond in BASELINE_PROMPTS:
                replies, regen = run_baseline(p, BASELINE_PROMPTS[cond], args.model, tutor_usage), 0
            else:
                replies, regen = run_proposed(p, cond == "proposed", tutor_usage)
            verdicts = [judge(p, r, args.judge_model, judge_usage) for r in replies]
        except Exception as e:  # API 오류도 결과로 남긴다
            return {"condition": cond, "problem_id": p["id"], "error": repr(e)}
        return {"condition": cond, "problem_id": p["id"], "unit": p.get("type", ""), "replies": replies,
                "verdicts": verdicts, "guard_regenerated": regen}

    tasks = [(c, p) for c in args.conditions for p in problems]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        dialogs = list(pool.map(run, tasks))

    summary = {}
    for cond in args.conditions:
        ds = [d for d in dialogs if d["condition"] == cond and "error" not in d]
        def rate(values):
            return round(sum(values) / len(values), 4) if values else None

        explicit = lambda v: v["level"] == "explicit"
        any_leak = lambda v: v["level"] != "none"
        by_unit = defaultdict(list)
        for d in ds:
            by_unit[d["unit"]].append(any(explicit(v) for v in d["verdicts"]))
        summary[cond] = {
            "dialogs": len(ds), "errors": sum(d["condition"] == cond and "error" in d for d in dialogs),
            "reply_explicit_rate": rate([explicit(v) for d in ds for v in d["verdicts"]]),
            "dialog_explicit_rate": rate([any(explicit(v) for v in d["verdicts"]) for d in ds]),
            "dialog_near_or_explicit_rate": rate([any(any_leak(v) for v in d["verdicts"]) for d in ds]),
            "explicit_by_turn": [rate([explicit(d["verdicts"][t]) for d in ds]) for t in range(1 + len(BEGGING))],
            "dialog_explicit_rate_by_unit": {u: rate(v) for u, v in by_unit.items()},
            "guard_regenerated": sum(d["guard_regenerated"] for d in ds),
        }
        print(f"[{cond}] " + json.dumps(summary[cond], ensure_ascii=False))

    meta = {"model": args.model, "judge_model": args.judge_model, "begging": BEGGING,
            "tutor_usage": tutor_usage.to_dict(), "judge_usage": judge_usage.to_dict(),
            "elapsed_sec": round(time.time() - started, 1)}
    print(json.dumps(meta, ensure_ascii=False))
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"exp3_leakage_{datetime.now():%Y%m%d-%H%M}.json"
    out.write_text(json.dumps({"meta": meta, "summary": summary, "dialogs": dialogs}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
