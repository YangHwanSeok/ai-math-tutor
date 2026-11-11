"""실험 4: 오개념 교정. 튜터와 대화한 가상 학생이 처음 보는 문제를 혼자 풀 수 있게 되는가.

시행 하나 = 오개념 하나 + 튜터링 문제 P1 + 확인 문제 2개(같은 오개념, 처음 보는 문제).
1. 사전   : 학생이 P1을 혼자 푼다. 오개념이 나타난 시행만 이후 단계를 진행한다.
2. 튜터링 : 같은 사전 풀이에서 조건별로 갈라져 튜터와 최대 MAX_TURNS번 주고받는다. P1을 맞히면 튜터의 마무리 한마디 후 끝낸다.
3. 확인   : 튜터 없이 확인 문제 2개를 푼다. SymPy로 정답 / 오개념 유지 / 다른 오답을 판정한다.

--first-turn answer(기본): 학생의 첫 제출은 답만 튜터에게 보인다(실제 학생 대부분이 답만 낸다).
--first-turn work        : 학생이 쓴 풀이 과정도 함께 보인다. 학생이 오개념을 말로 드러내므로 진단 없이도 원인을 알 수 있다.

조건 (튜터 모델은 모두 같다)
- retry_only            : 대조. 맞았는지만 알려 주고 "다시 풀어 볼래?"만 반복
- chatbot               : 일반 챗봇
- prompt_tutor          : 프롬프트형 튜터 (실험 3과 같은 지시문, 학습 모드형)
- prompt_tutor_solution : 프롬프트형 튜터 + 교사가 준 정답 (Bastani 외의 GPT Tutor, Khanmigo 방식)
- proposed              : 제안 튜터 (SymPy 채점·오개념 진단 + 학습자 모델, 정답 미제공)
- bridge                : 기존 LLM 교정 기법 (Wang 외, NAACL 2024). 오류 유형·교정 전략·의도를 LLM이 먼저 정한 뒤 응답
- proposed_no_diag      : 제거 실험. 제안 튜터에서 진단만 끔 (정답/오답 여부만 전달)

--reuse-pre <결과 파일>: 기존 실행의 사전 풀이를 그대로 써서 새 조건만 추가로 돌린다(기존 조건과 짝지은 비교 유지).

실행: python -m experiments.exp4_tutoring
      python -m experiments.exp4_tutoring --misconceptions <오개념 id ...>  (통계·도형 등)
정답 누설은 규칙 기반 검사(leaks_answer)로 세며, 수 하나가 답인 문제(value)는 자료 속 수와 구분할 수 없어 세지 않는다.
"""
import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from experiments.exp3_leakage import BASELINE_PROMPTS
from experiments.student_sim import (MISCONCEPTIONS, UNITS, as_history, build_problem, classify, correct_answer,
                                     problem_message, student_completer)
from mathtutor import llm
from mathtutor.learner import LearnerModel
from mathtutor.tutor import MESSAGE_SCHEMA, TutorSession, leaks_answer, llm_extractor, llm_responder

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CONDITIONS = ["retry_only", "chatbot", "prompt_tutor", "prompt_tutor_solution", "bridge", "proposed",
              "proposed_no_diag"]
# nonzero_product는 예비 시험에서 설명 없이도 67%가 고쳐져 제외한다
DEFAULT_MISCONCEPTIONS = ["factor_sign", "root_sign", "divide_by_x", "positive_sqrt_only"]
POST_TEST_COUNT = 2
POST_TEST_INTRO = "이제 선생님 없이 혼자 풀어 보는 확인 문제야."

SOLUTION_PROMPT = BASELINE_PROMPTS["prompt_tutor"] + """
- 아래 [참고 정답]은 교사가 준 것이다. 학생 답이 맞는지 판단할 때 쓰고, 학생에게 그대로 말하지 않는다.
[참고 정답] {answer}"""


class RetryOnlyTutor:
    def __init__(self, problem: dict):
        self.problem = problem

    def reply(self, student_text: str, answer: str) -> str:
        return "맞았어!" if answer and classify(self.problem, answer) == "correct" else "틀렸어. 다시 풀어 볼래?"


class PromptTutor:
    """지시문만으로 동작하는 LLM 튜터. 대화 기록 전체를 그대로 넘긴다."""
    def __init__(self, problem: dict, prompt: str, model: str, usage: llm.Usage):
        self.problem = problem
        self.respond = llm.json_completer(model, prompt, MESSAGE_SCHEMA, "reply", "low", usage)
        self.history: list[dict] = []

    def reply(self, student_text: str, answer: str) -> str:
        if not self.history:
            student_text = f"문제: {self.problem['statement']}\n{student_text}"
        self.history.append({"role": "user", "content": student_text})
        message = self.respond(self.history)["message"]
        self.history.append({"role": "assistant", "content": message})
        return message


# Bridge (Wang 외, NAACL 2024): 전문 교사가 실수를 교정할 때 내리는 세 가지 판단을 LLM이 먼저 정하고(Self 조건),
# 그 판단에 맞춰 응답을 만든다. 범주 이름은 논문 4.2절을 그대로 쓴다.
BRIDGE_ERROR_TYPES = {
    "guess": "학생이 이해하지 못했거나 답을 찍은 것 같다",
    "misinterpret": "학생이 문제를 잘못 해석했다",
    "careless": "개념은 맞게 적용했지만 사소한 계산 실수를 했다",
    "right-idea": "생각의 방향은 맞지만 개념을 제대로 적용하지 못했다",
    "imprecise": "답이 충분히 정확하지 않거나 형식 문제다",
    "not-sure": "확실하지 않지만 진단을 시도한다",
    "N/A": "해당 없음",
}
BRIDGE_STRATEGIES = ["Explain a concept", "Ask a question", "Provide a hint", "Provide a strategy",
                     "Provide a worked example", "Provide a minor correction", "Provide a similar problem",
                     "Simplify the question", "Affirm the correct answer", "Encourage the student", "Other"]
BRIDGE_INTENTIONS = ["Motivate the student", "Get the student to elaborate their answer", "Correct the mistake",
                     "Hint at the mistake", "Clarify the misunderstanding",
                     "Help the student understand the lesson topic or solution strategy", "Diagnose the mistake",
                     "Support the student in their thinking or problem-solving", "Explain the mistake",
                     "Signal to the student that they have solved or not solved the problem", "Other"]
BRIDGE_DECISION_PROMPT = """너는 중학생을 가르치는 경험 많은 수학 교사다. 학생의 마지막 답을 보고, 다음 응답을 하기 전에 세 가지를 정한다.
1. error_type: 학생 오류의 유형
{error_types}
2. strategy: 이번 응답에서 쓸 교정 전략
3. intention: 그 전략으로 이루려는 의도
reason에는 그렇게 판단한 근거를 한 문장으로 쓴다.""".format(
    error_types="\n".join(f"   - {k}: {v}" for k, v in BRIDGE_ERROR_TYPES.items()))
BRIDGE_DECISION_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"},
                   "error_type": {"type": "string", "enum": list(BRIDGE_ERROR_TYPES)},
                   "strategy": {"type": "string", "enum": BRIDGE_STRATEGIES},
                   "intention": {"type": "string", "enum": BRIDGE_INTENTIONS}},
    "required": ["reason", "error_type", "strategy", "intention"],
    "additionalProperties": False,
}
BRIDGE_RESPONSE_PROMPT = BASELINE_PROMPTS["prompt_tutor"] + """
- 아래 [교사의 판단]에 따라 응답한다.
[교사의 판단] 오류 유형: {error_type} / 전략: {strategy} / 의도: {intention} / 근거: {reason}"""


class BridgeTutor:
    """기존 LLM 교정 기법: 오류 유형·전략·의도를 LLM이 스스로 정한 뒤 응답한다. 채점·진단 도구는 없다."""
    def __init__(self, problem: dict, model: str, usage: llm.Usage):
        self.problem, self.model, self.usage = problem, model, usage
        self.decide = llm.json_completer(model, BRIDGE_DECISION_PROMPT, BRIDGE_DECISION_SCHEMA, "decision", "low", usage)
        self.history: list[dict] = []
        self.last_info: dict | None = None

    def reply(self, student_text: str, answer: str) -> str:
        if not self.history:
            student_text = f"문제: {self.problem['statement']}\n{student_text}"
        self.history.append({"role": "user", "content": student_text})
        decision = self.decide(self.history)
        respond = llm.json_completer(self.model, BRIDGE_RESPONSE_PROMPT.format(**decision), MESSAGE_SCHEMA, "reply",
                                     "low", self.usage)
        message = respond(self.history)["message"]
        self.history.append({"role": "assistant", "content": message})
        self.last_info = decision
        return message


class ProposedTutor:
    def __init__(self, problem: dict, show_diagnosis: bool, usage: llm.Usage):
        self.session = TutorSession(problem, LearnerModel(), llm_extractor(usage), llm_responder(usage),
                                    show_diagnosis=show_diagnosis)

    def reply(self, student_text: str, answer: str) -> str:
        return self.session.handle(student_text)


def make_tutor(cond: str, problem: dict, model: str, usage: llm.Usage):
    if cond == "retry_only":
        return RetryOnlyTutor(problem)
    if cond == "chatbot":
        return PromptTutor(problem, BASELINE_PROMPTS["chatbot"], model, usage)
    if cond == "prompt_tutor":
        return PromptTutor(problem, BASELINE_PROMPTS["prompt_tutor"], model, usage)
    if cond == "prompt_tutor_solution":
        return PromptTutor(problem, SOLUTION_PROMPT.format(answer=correct_answer(problem)), model, usage)
    if cond == "bridge":
        return BridgeTutor(problem, model, usage)
    return ProposedTutor(problem, show_diagnosis=cond == "proposed", usage=usage)


def run_pre(mid: str, index: int, student_model: str, effort: str, usage: llm.Usage) -> dict:
    student = student_completer(mid, student_model, effort, usage)
    p1 = build_problem(mid, index)
    history = [{"role": "user", "content": problem_message(p1)}]
    reply = student(history)
    return {"misconception": mid, "index": index, "p1": p1["id"], "pre_reply": reply,
            "pre_label": classify(p1, reply["answer"]), "history": history + [as_history(reply)]}


def reuse_pres(path: Path, args) -> list[dict]:
    """기존 결과 파일의 사전 풀이를 불러온다. 대화 기록은 문제와 학생 응답으로 똑같이 다시 만든다."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["meta"]["first_turn"] != args.first_turn:
        raise SystemExit(f"--first-turn이 기존 실행({data['meta']['first_turn']})과 다릅니다")
    args.misconceptions = data["meta"]["misconceptions"]
    pres = []
    for p in data["pre"]:
        if "error" not in p:
            p1 = build_problem(p["misconception"], p["index"])
            p = {**p, "history": [{"role": "user", "content": problem_message(p1)}, as_history(p["pre_reply"])]}
        pres.append(p)
    return pres


def retry_errors(path: Path, args, usage: dict[str, llm.Usage]) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data["meta"]
    args.first_turn, args.max_turns = meta["first_turn"], meta["max_turns"]
    pres = {(p["p1"], p["rep"]): p for p in reuse_pres(path, args) if "error" not in p}
    failed = [i for i, d in enumerate(data["dialogs"]) if "error" in d]

    def redo(i):
        d = data["dialogs"][i]
        try:
            return run_branch(pres[(d["p1"], d["rep"])], d["condition"], args, usage)
        except Exception as e:  # 다시 실패하면 오류로 남겨 두고 다음 재실행에서 처리한다
            return {**d, "error": repr(e)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        redone = list(pool.map(redo, failed))
    for i, d in zip(failed, redone):
        data["dialogs"][i] = d
    data["summary"] = summarize(data["dialogs"], meta["misconceptions"])
    meta["retried_errors"] = meta.get("retried_errors", 0) + sum("error" not in d for d in redone)
    print(f"재실행 성공 {sum('error' not in d for d in redone)} / {len(failed)}")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data["summary"], ensure_ascii=False, indent=2))
    print(f"오류 {len(failed)}건 재실행, 저장: {path}")


def run_branch(pre: dict, cond: str, args, usage: dict[str, llm.Usage]) -> dict:
    mid, index = pre["misconception"], pre["index"]
    student = student_completer(mid, args.student_model, args.student_effort, usage["student"])
    p1 = build_problem(mid, index)
    tutor = make_tutor(cond, p1, args.tutor_model, usage["tutor"])
    history = list(pre["history"])
    answer = pre["pre_reply"]["answer"]
    text = history[-1]["content"] if args.first_turn == "work" else f"답: {answer}"
    transcript = [{"role": "student", "text": text, "answer": answer, "label": pre["pre_label"]}]

    solved_at = None
    for turn in range(1, args.max_turns + 1):
        message = tutor.reply(text, answer)
        transcript.append({"role": "tutor", "text": message, "leak": leaks_answer(message, p1, 0),
                           **({"decision": tutor.last_info} if getattr(tutor, "last_info", None) else {})})
        history.append({"role": "user", "content": message})
        reply = student(history)
        history.append(as_history(reply))
        text, answer = history[-1]["content"], reply["answer"]
        label = classify(p1, answer) if answer else "no_answer"
        transcript.append({"role": "student", "text": text, "answer": answer, "label": label})
        if label == "correct":
            solved_at = turn
            closing = tutor.reply(text, answer)
            transcript.append({"role": "tutor", "text": closing, "leak": False})
            history.append({"role": "user", "content": closing})
            break

    n = len(MISCONCEPTIONS[mid]["problems"])
    post = []
    for k in range(1, POST_TEST_COUNT + 1):
        problem = build_problem(mid, (index + k) % n)
        intro = POST_TEST_INTRO + "\n" if k == 1 else ""
        history.append({"role": "user", "content": intro + problem_message(problem)})
        reply = student(history)
        history.append(as_history(reply))
        post.append({"problem": problem["id"], "answer": reply["answer"], "message": reply["message"],
                     "label": classify(problem, reply["answer"])})

    return {"condition": cond, "misconception": mid, "p1": p1["id"], "rep": pre["rep"], "solved_at": solved_at,
            "tutor_leaked": (any(t.get("leak") for t in transcript if t["role"] == "tutor")
                             if p1["steps"][0]["check"]["type"] != "value" else None),
            "transcript": transcript, "post": post}


def _rate(values: list) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _bootstrap_ci(per_trial: list[float], seed: int = 0, n: int = 2000) -> list[float] | None:
    if not per_trial:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(per_trial, k=len(per_trial))) / len(per_trial) for _ in range(n))
    return [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)]


def summarize(dialogs: list[dict], misconceptions: list[str]) -> dict:
    summary = {}
    for cond in CONDITIONS:
        ds = [d for d in dialogs if d["condition"] == cond and "error" not in d]
        if not ds:
            continue
        post_correct = [sum(p["label"] == "correct" for p in d["post"]) / len(d["post"]) for d in ds]
        solved = [d["solved_at"] for d in ds if d["solved_at"] is not None]
        summary[cond] = {
            "dialogs": len(ds),
            "errors": sum(d["condition"] == cond and "error" in d for d in dialogs),
            "post_correct_rate": _rate(post_correct),
            "post_correct_ci95": _bootstrap_ci(post_correct),
            "post_all_correct_rate": _rate([all(p["label"] == "correct" for p in d["post"]) for d in ds]),
            "post_bug_rate": _rate([p["label"] == "bug" for d in ds for p in d["post"]]),
            "post_other_rate": _rate([p["label"] in ("other", "unparseable") for d in ds for p in d["post"]]),
            "p1_solved_rate": _rate([d["solved_at"] is not None for d in ds]),
            "avg_turns_to_solve": _rate(solved),
            "tutor_leak_rate": _rate([d["tutor_leaked"] for d in ds if d["tutor_leaked"] is not None]),
            "post_correct_by_misconception": {
                mid: _rate([pc for d, pc in zip(ds, post_correct) if d["misconception"] == mid]) for mid in misconceptions},
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--misconceptions", nargs="+", default=DEFAULT_MISCONCEPTIONS)
    parser.add_argument("--tutor-model", default="gpt-5.4")
    parser.add_argument("--student-model", default="gpt-5.4")
    parser.add_argument("--student-effort", default="none")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--first-turn", choices=["answer", "work"], default="answer")
    parser.add_argument("--limit", type=int, default=None, help="오개념마다 앞의 문제 몇 개만 쓸지 (시험 실행용)")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--reuse-pre", default=None, help="기존 결과 파일의 사전 풀이를 재사용")
    parser.add_argument("--retry-errors", default=None, help="결과 파일에서 API 오류로 끝난 대화만 다시 실행해 채운다")
    args = parser.parse_args()
    os.environ["TUTOR_MODEL"] = args.tutor_model  # 제안 튜터(TutorSession)도 같은 모델을 쓰게 한다

    usage = {"student": llm.Usage(), "tutor": llm.Usage()}
    if args.retry_errors:
        retry_errors(Path(args.retry_errors), args, usage)
        return
    started = time.time()

    pre_tasks = [(mid, i, rep) for mid in args.misconceptions
                 for i in range(len(MISCONCEPTIONS[mid]["problems"]))[: args.limit] for rep in range(args.reps)]

    def pre(task):
        mid, i, rep = task
        try:
            return {"rep": rep, **run_pre(mid, i, args.student_model, args.student_effort, usage["student"])}
        except Exception as e:  # API 오류도 결과로 남긴다
            return {"rep": rep, "misconception": mid, "index": i, "error": repr(e)}

    if args.reuse_pre:
        pres = reuse_pres(Path(args.reuse_pre), args)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pres = list(pool.map(pre, pre_tasks))
    shown = [p for p in pres if p.get("pre_label") == "bug"]

    def branch(task):
        p, cond = task
        try:
            return run_branch(p, cond, args, usage)
        except Exception as e:
            return {"condition": cond, "misconception": p["misconception"], "p1": p["p1"], "rep": p["rep"],
                    "error": repr(e)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        dialogs = list(pool.map(branch, [(p, c) for p in shown for c in args.conditions]))

    summary = summarize(dialogs, args.misconceptions)
    meta = {"tutor_model": args.tutor_model, "student_model": args.student_model,
            "student_effort": args.student_effort, "reps": args.reps, "max_turns": args.max_turns,
            "first_turn": args.first_turn, "conditions": args.conditions, "reused_pre_from": args.reuse_pre,
            "misconceptions": args.misconceptions, "pre_trials": len(pres),
            "pre_bug_rate": _rate([p.get("pre_label") == "bug" for p in pres]),
            "pre_errors": sum("error" in p for p in pres),
            "usage": {k: u.to_dict() for k, u in usage.items()}, "elapsed_sec": round(time.time() - started, 1)}
    print(json.dumps({"meta": meta, "summary": summary}, ensure_ascii=False, indent=2))
    RESULTS_DIR.mkdir(exist_ok=True)
    units = sorted({u for u, mids in UNITS.items() for m in args.misconceptions if m in mids})
    tag = "-".join(units) + ("" if args.conditions == CONDITIONS else "_" + "-".join(args.conditions))
    out = RESULTS_DIR / f"exp4_tutoring_{args.first_turn}_{tag}_{datetime.now():%Y%m%d-%H%M%S}.json"
    pres_out = [{k: v for k, v in p.items() if k != "history"} for p in pres]
    out.write_text(json.dumps({"meta": meta, "summary": summary, "pre": pres_out, "dialogs": dialogs},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
