"""예비 시험: 가상 학생이 심어 둔 오개념을 일관되게 유지하는가.

시행 하나는 같은 오개념의 문제 P1, P2로 이루어진다.
1. 사전      : P1을 혼자 푼다.                      → 오개념이 나타나야 한다 (유지율)
2. 대조      : "틀렸어. 다시 풀어 볼래?"만 듣고 P1을 다시 풀고, 이어서 P2를 푼다.
               → 설명 없이 저절로 고쳐지면 안 된다 (자가 교정률이 낮아야 함)
3. 모범 설명 : 오개념을 겨냥한 설명을 듣고 P2를 푼다.   → 고쳐져야 한다 (교정 가능률이 높아야 함)
4. 정답만    : P1의 정답만 듣고 P2를 푼다.             → 답을 바로 주는 튜터. 3보다 덜 고쳐져야
               본 실험에서 '설명의 질'에 따른 차이가 드러난다.
2~4는 1의 대화에서 갈라진다.

실행: python -m experiments.exp4_student_pilot --model gpt-5.4 --effort none --units statistics geometry
"""
import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from experiments.student_sim import (MISCONCEPTIONS, UNITS, as_history, build_problem, classify, correct_answer,
                                     problem_message, student_completer)
from mathtutor import llm

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RETRY = "틀렸어. 다시 풀어 볼래?"


def run_trial(mid: str, index: int, model: str, effort: str | None, usage: llm.Usage) -> dict:
    student = student_completer(mid, model, effort, usage)
    n = len(MISCONCEPTIONS[mid]["problems"])
    p1, p2 = build_problem(mid, index), build_problem(mid, (index + 1) % n)

    def ask(history: list[dict], problem: dict) -> tuple[dict, str]:
        reply = student(history)
        return reply, classify(problem, reply["answer"])

    base = [{"role": "user", "content": problem_message(p1)}]
    pre, pre_label = ask(base, p1)

    retry_hist = base + [as_history(pre), {"role": "user", "content": RETRY}]
    retry, retry_label = ask(retry_hist, p1)
    control_hist = retry_hist + [as_history(retry), {"role": "user", "content": problem_message(p2)}]
    control, control_label = ask(control_hist, p2)

    explained_hist = base + [as_history(pre), {"role": "user", "content": MISCONCEPTIONS[mid]["explanation"]
                                               + "\n이제 다음 문제를 풀어 볼래?\n" + problem_message(p2)}]
    explained, explained_label = ask(explained_hist, p2)

    answer_hist = base + [as_history(pre), {"role": "user", "content": f"틀렸어. 정답은 {correct_answer(p1)}이야."
                                            + "\n이제 다음 문제를 풀어 볼래?\n" + problem_message(p2)}]
    answered, answered_label = ask(answer_hist, p2)

    return {"misconception": mid, "p1": p1["id"], "p2": p2["id"],
            "labels": {"pre": pre_label, "retry": retry_label, "control_p2": control_label,
                       "explained_p2": explained_label, "answer_only_p2": answered_label},
            "replies": {"pre": pre, "retry": retry, "control_p2": control, "explained_p2": explained,
                        "answer_only_p2": answered}}


def rates(labels: list[str]) -> dict:
    counts = Counter(labels)
    return {k: round(counts[k] / len(labels), 4) for k in ("bug", "correct", "other", "unparseable")} if labels else {}


def summarize(trials: list[dict], misconceptions: list[str]) -> dict:
    ok = [t for t in trials if "error" not in t]
    shown = [t for t in ok if t["labels"]["pre"] == "bug"]  # 사전에 오개념이 나타난 시행만 이후 단계를 해석할 수 있다
    phase = lambda ts, key: rates([t["labels"][key] for t in ts])
    return {
        "trials": len(ok), "errors": len(trials) - len(ok),
        "pre": phase(ok, "pre"),
        "given_pre_bug": {"n": len(shown), "retry": phase(shown, "retry"), "control_p2": phase(shown, "control_p2"),
                          "answer_only_p2": phase(shown, "answer_only_p2"), "explained_p2": phase(shown, "explained_p2")},
        "by_misconception": {mid: {"pre_bug": phase([t for t in ok if t["misconception"] == mid], "pre").get("bug"),
                                   "control_p2_correct": phase([t for t in shown if t["misconception"] == mid],
                                                               "control_p2").get("correct"),
                                   "answer_only_p2_correct": phase([t for t in shown if t["misconception"] == mid],
                                                                   "answer_only_p2").get("correct"),
                                   "explained_p2_correct": phase([t for t in shown if t["misconception"] == mid],
                                                                 "explained_p2").get("correct")}
                             for mid in misconceptions},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--effort", default="none")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--units", nargs="+", choices=list(UNITS), default=["algebra"])
    parser.add_argument("--only", nargs="+", default=None, help="단원 대신 오개념을 직접 지정")
    args = parser.parse_args()

    usage, started = llm.Usage(), time.time()

    def run(task):
        mid, index, rep = task
        try:
            return {"rep": rep, **run_trial(mid, index, args.model, args.effort, usage)}
        except Exception as e:  # API 오류도 결과로 남긴다
            return {"rep": rep, "misconception": mid, "p1": f"{mid}-{index}", "error": repr(e)}

    misconceptions = [mid for unit in args.units for mid in UNITS[unit]]
    if args.only:
        misconceptions = args.only
    tasks = [(mid, i, rep) for mid in misconceptions for i in range(len(MISCONCEPTIONS[mid]["problems"]))
             for rep in range(args.reps)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        trials = list(pool.map(run, tasks))

    summary = summarize(trials, misconceptions)
    meta = {"model": args.model, "effort": args.effort, "reps": args.reps, "units": args.units, "retry_message": RETRY,
            "usage": usage.to_dict(), "elapsed_sec": round(time.time() - started, 1)}
    print(json.dumps({"meta": meta, "summary": summary}, ensure_ascii=False, indent=2))
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"exp4_pilot_{'-'.join(args.units)}_{args.model}_{args.effort}_{datetime.now():%Y%m%d-%H%M}.json"
    out.write_text(json.dumps({"meta": meta, "summary": summary, "trials": trials}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
