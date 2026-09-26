"""실험 1: SymPy 검증·재생성 전후의 풀이 정답률 비교.

실행: python -m experiments.exp1_solver_verification --model gpt-5.4-mini --effort low --trials 3
"""
import argparse
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from mathtutor.llm import Usage
from mathtutor.problems import load_problems
from mathtutor.solver import llm_solver, solve_with_verification

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _attempt_to_dict(attempt) -> dict:
    return {
        "steps": attempt.steps,
        "results": [{"severity": r.severity.value, "error": r.error.value,
                     "diagnosis": r.diagnosis, "detail": r.detail} for r in attempt.results],
        "correct": attempt.correct,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("SOLVER_MODEL"))
    parser.add_argument("--effort", default="low")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    usage = Usage()
    complete = llm_solver(args.model, args.effort, usage)
    runs, started = [], time.time()
    for problem in load_problems().values():
        for trial in range(args.trials):
            try:
                record = solve_with_verification(problem, complete, args.max_attempts)
            except Exception as e:  # API·파싱 오류도 실험 결과로 기록한다
                runs.append({"problem_id": problem["id"], "trial": trial, "api_error": repr(e)})
                print(f"{problem['id']} #{trial}: API 오류 {e!r}")
                continue
            runs.append({
                "problem_id": problem["id"], "trial": trial,
                "first_try_correct": record.first_try_correct, "verified": record.verified,
                "attempts": [_attempt_to_dict(a) for a in record.attempts],
            })
            print(f"{problem['id']} #{trial}: 첫 시도 {'O' if record.first_try_correct else 'X'}, "
                  f"최종 {'O' if record.verified else 'X'} ({len(record.attempts)}회)")

    ok = [r for r in runs if "api_error" not in r]
    first_errors = Counter(res["error"] for r in ok for res in r["attempts"][0]["results"] if res["error"] != "없음")
    summary = {
        "model": args.model, "effort": args.effort, "trials": args.trials, "max_attempts": args.max_attempts,
        "runs": len(runs), "api_errors": len(runs) - len(ok),
        "first_try_accuracy": sum(r["first_try_correct"] for r in ok) / len(ok),
        "verified_accuracy": sum(r["verified"] for r in ok) / len(ok),
        "wrong_without_verification": sum(not r["first_try_correct"] for r in ok),
        "blocked_after_max_attempts": sum(not r["verified"] for r in ok),
        "avg_attempts": sum(len(r["attempts"]) for r in ok) / len(ok),
        "first_try_error_types": dict(first_errors),
        "usage": asdict(usage), "elapsed_sec": round(time.time() - started, 1),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"exp1_{args.model}_{args.effort}_{datetime.now():%Y%m%d-%H%M}.json"
    out.write_text(json.dumps({"summary": summary, "runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
