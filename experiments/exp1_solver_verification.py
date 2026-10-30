"""실험 1: 풀이 검증 방식 비교.

방식
- sympy : 제안 방식. SymPy로 단계별 답을 검증하고, 실패 내용을 알려주며 다시 생성한다.
- self  : 기존 방식(Self-Refine형 자기 검증). 같은 LLM이 자기 풀이를 검토하고, 검토 의견으로 다시 생성한다.

두 방식 모두 최대 max_attempts번 안에 통과(검증 성공 / 검토자가 맞다고 판단)하지 못하면 풀이를 내보내지 않는다(차단).
self 방식에서도 각 시도의 실제 정오는 SymPy로 채점하지만, 그 결과는 측정에만 쓰고 LLM에게 알려주지 않는다.

실행: python -m experiments.exp1_solver_verification --model gpt-5.4-nano --effort none --method self
"""
import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from mathtutor import llm
from mathtutor.llm import Usage
from mathtutor.problems import CASE_STUDY_IDS, load_problems
from mathtutor.solver import Attempt, build_prompt, llm_solver, solve_with_verification, verify

RESULTS_DIR = Path(__file__).resolve().parent / "results"

REVIEW_INSTRUCTIONS = """너는 중학교 수학 풀이의 검토자다.
- 문제와 단계별 풀이를 보고, 각 단계의 answer가 맞는지 직접 다시 계산해서 확인한다.
- answer가 지정된 형식을 지키는지도 확인한다.
- 모든 단계가 맞으면 correct를 true로 한다.
- 틀린 단계가 있으면 correct를 false로 하고, feedback에 몇 단계가 왜 틀렸는지 쓴다."""

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}, "feedback": {"type": "string"}},
    "required": ["correct", "feedback"],
    "additionalProperties": False,
}


def _attempt_to_dict(attempt: Attempt, review: dict | None = None) -> dict:
    out = {
        "steps": attempt.steps,
        "results": [{"severity": r.severity.value, "error": r.error.value,
                     "diagnosis": r.diagnosis, "detail": r.detail} for r in attempt.results],
        "correct": attempt.correct,
    }
    if review is not None:
        out["review"] = review
    return out


def solve_with_self_check(problem: dict, complete: llm.JsonCompleter, review: llm.JsonCompleter,
                          max_attempts: int = 3) -> dict:
    """같은 LLM이 풀이를 검토하고, 틀렸다고 판단하면 검토 의견을 주고 다시 풀게 한다."""
    prompt = build_prompt(problem)
    messages = [{"role": "user", "content": prompt}]
    attempts, accepted = [], False
    for _ in range(max_attempts):
        steps = complete(messages)["steps"]
        solution = json.dumps({"steps": steps}, ensure_ascii=False)
        verdict = review([{"role": "user", "content": f"{prompt}\n\n검토할 풀이:\n{solution}"}])
        attempts.append(_attempt_to_dict(Attempt(steps, verify(problem, steps)), verdict))
        if verdict["correct"]:
            accepted = True
            break
        messages += [
            {"role": "assistant", "content": solution},
            {"role": "user", "content": f"검토 결과: {verdict['feedback']}\n검토 내용을 반영해 전체 풀이를 다시 제출하라."},
        ]
    return {"first_try_correct": attempts[0]["correct"], "accepted": accepted,
            "delivered_correct": accepted and attempts[-1]["correct"],
            "delivered_wrong": accepted and not attempts[-1]["correct"], "attempts": attempts}


def solve_with_sympy(problem: dict, complete: llm.JsonCompleter, max_attempts: int) -> dict:
    record = solve_with_verification(problem, complete, max_attempts)
    return {"first_try_correct": record.first_try_correct, "accepted": record.verified,
            "delivered_correct": record.verified, "delivered_wrong": False,
            "attempts": [_attempt_to_dict(a) for a in record.attempts]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("SOLVER_MODEL"))
    parser.add_argument("--effort", default="low")
    parser.add_argument("--method", choices=["sympy", "self"], default="sympy")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    usage = Usage()
    complete = llm_solver(args.model, args.effort, usage)
    review = llm.json_completer(args.model, REVIEW_INSTRUCTIONS, REVIEW_SCHEMA, "review", args.effort, usage)
    started = time.time()

    def run(task):
        problem, trial = task
        try:
            if args.method == "self":
                record = solve_with_self_check(problem, complete, review, args.max_attempts)
            else:
                record = solve_with_sympy(problem, complete, args.max_attempts)
        except Exception as e:  # API·파싱 오류도 실험 결과로 기록한다
            print(f"{problem['id']} #{trial}: API 오류 {e!r}")
            return {"problem_id": problem["id"], "trial": trial, "api_error": repr(e)}
        print(f"{problem['id']} #{trial}: 첫 시도 {'O' if record['first_try_correct'] else 'X'}, "
              f"{'통과' if record['accepted'] else '차단'}"
              f"{'(오답 전달)' if record['delivered_wrong'] else ''} ({len(record['attempts'])}회)")
        return {"problem_id": problem["id"], "trial": trial, **record}

    tasks = [(p, t) for p in load_problems().values() if p["id"] not in CASE_STUDY_IDS for t in range(args.trials)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(run, tasks))

    ok = [r for r in runs if "api_error" not in r]
    n = len(ok)
    first_errors = Counter(res["error"] for r in ok for res in r["attempts"][0]["results"] if res["error"] != "없음")
    summary = {
        "model": args.model, "effort": args.effort, "method": args.method,
        "trials": args.trials, "max_attempts": args.max_attempts,
        "runs": len(runs), "api_errors": len(runs) - n,
        "first_try_accuracy": sum(r["first_try_correct"] for r in ok) / n,
        "delivered_correct_rate": sum(r["delivered_correct"] for r in ok) / n,
        "delivered_wrong_rate": sum(r["delivered_wrong"] for r in ok) / n,
        "blocked_rate": sum(not r["accepted"] for r in ok) / n,
        "wrong_without_verification": sum(not r["first_try_correct"] for r in ok),
        "delivered_wrong": sum(r["delivered_wrong"] for r in ok),
        "blocked_after_max_attempts": sum(not r["accepted"] for r in ok),
        "avg_attempts": sum(len(r["attempts"]) for r in ok) / n,
        "first_try_error_types": dict(first_errors),
    }
    if args.method == "self":
        reviewed = [a for r in ok for a in r["attempts"]]
        wrong = [a for a in reviewed if not a["correct"]]
        right = [a for a in reviewed if a["correct"]]
        summary["review"] = {
            "reviewed_attempts": len(reviewed),
            # 틀린 풀이를 맞다고 통과시킨 비율 / 맞는 풀이를 틀렸다고 한 비율
            "false_accept_rate": sum(a["review"]["correct"] for a in wrong) / len(wrong) if wrong else None,
            "false_reject_rate": sum(not a["review"]["correct"] for a in right) / len(right) if right else None,
            "first_correct_but_final_wrong": sum(r["first_try_correct"] and not r["attempts"][-1]["correct"] for r in ok),
        }
    summary["usage"] = usage.to_dict()
    summary["elapsed_sec"] = round(time.time() - started, 1)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"exp1_{args.method}_{args.model}_{args.effort}_{datetime.now():%Y%m%d-%H%M}.json"
    out.write_text(json.dumps({"summary": summary, "runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
