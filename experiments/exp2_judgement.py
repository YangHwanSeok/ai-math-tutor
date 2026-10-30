"""실험 2: 학생 답 판정·진단 정확도 비교 (SymPy 채점기 vs LLM 채점).

조건
- sympy   : 제안 기법. 채점 엔진(check_step)으로 판정
- llm     : 기존 방식. LLM이 정답 없이 스스로 판단하며 채점
- llm_ref : 기존 방식(강화). LLM에게 정답을 함께 주고 채점

평가 데이터는 문제 은행에서 자동 생성한다. 정답·오답 라벨은 채점 엔진을 쓰지 않고
SymPy 값 비교로 따로 확인해, 제안 기법에 유리하게 순환되지 않도록 한다.

실행: python -m experiments.exp2_judgement --conditions sympy llm llm_ref
"""
import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import sympy as sp

from mathtutor import llm
from mathtutor.checker import ErrorType, Severity, check_step, parse_math, problem_symbols, solve_condition
from mathtutor.problems import CASE_STUDY_IDS, load_problems

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SIGN, OTHER, NONE = "부호 실수", "기타 오류", "없음"


# ---------------------------------------------------------------- 평가 데이터 생성

def _s(expr: sp.Expr) -> str:
    return str(expr).replace("**", "^")


def _canonical(problem: dict, check: dict):
    """단계의 정답을 SymPy 값으로 돌려준다 (유형별 형태가 다름)."""
    symbols = problem_symbols(problem)
    kind = check["type"]
    if kind == "factor":
        return parse_math(check["expected"], symbols)
    if kind == "equivalent":
        return sp.expand(parse_math(check["target"], symbols))
    if kind == "roots":
        return [parse_math(r, symbols) for r in check["expected"]]
    if kind == "value":
        return parse_math(check["expected"], symbols)
    if kind == "assignment":
        solution = solve_condition(problem)[0]
        asked = check.get("ask", problem["unknowns"])
        return {k: v for k, v in solution.items() if k in asked}
    raise ValueError(kind)


def _unit(problem: dict) -> str:
    st = problem["statement"]
    if problem.get("type") != "geometry":
        return ""
    return "°" if "°" in st else "cm²" if "cm²" in st else "cm" if "cm" in st else ""


def _pretty_sqrt(text: str) -> str:
    return re.sub(r"sqrt\((\d+)\)", r"√\1", text).replace("*√", "√")


def _correct_variants(problem: dict, check: dict, truth) -> list[str]:
    kind = check["type"]
    if kind == "factor":
        parts = re.findall(r"\([^()]*\)", check["expected"])
        variants = [check["expected"], f"{problem['expression']}={check['expected']}"]
        if len(parts) == 2 and "".join(parts) == check["expected"]:
            variants.append(parts[1] + parts[0])
        return variants
    if kind == "equivalent":
        var = problem_symbols(problem)[problem.get("variable", "x")]
        return [_s(truth), _s(sp.collect(truth, var)), f"{check['target']}={_s(truth)}"]
    if kind == "roots":
        strs = [_s(r) for r in truth]
        variants = [", ".join(strs), " 또는 ".join(f"x={r}" for r in reversed(strs))]
        if len(truth) == 2 and _s(truth[0]).count("sqrt"):
            variants.append(_pretty_sqrt(", ".join(strs)))
        return variants
    if kind == "value":
        text = _s(truth)
        variants = [text]
        if "sqrt" in text:
            variants.append(_pretty_sqrt(text))
        if truth.is_Rational and not truth.is_Integer and 10 ** 6 % truth.q == 0:
            variants.append(str(float(truth)).rstrip("0"))
        if _unit(problem):
            variants.append(text + _unit(problem))
        return variants
    if kind == "assignment":
        items = [f"{k}={_s(v)}" for k, v in truth.items()]
        variants = [", ".join(items)]
        if len(items) > 1:
            variants.append("이고 ".join(reversed(items)))
            if problem.get("type") == "system":
                variants.append("(" + ", ".join(_s(truth[n]) for n in problem["unknowns"]) + ")")
        else:
            variants.append(_s(next(iter(truth.values()))))
        return variants
    raise ValueError(kind)


def _wrong_variants(problem: dict, check: dict, truth) -> list[tuple[str, str]]:
    """(답, 범주) 목록. 범주: sign(부호 실수), off(값 하나가 1만큼 다름)."""
    kind = check["type"]
    if kind == "factor":
        var = problem_symbols(problem)[problem.get("variable", "x")]
        flipped = sp.factor(sp.expand(truth).subs(var, -var))
        if sp.Poly(flipped, var).LC() < 0:
            flipped = sp.factor(-flipped)
        return [(_s(flipped), "sign")]
    if kind == "equivalent":
        return [(_s(truth + 1), "off")]
    if kind == "roots":
        return [(", ".join(_s(-r) for r in truth), "sign"),
                (", ".join(_s(r + 1) if i == 0 else _s(r) for i, r in enumerate(truth)), "off")]
    if kind == "value":
        return ([(_s(-truth), "sign")] if truth != 0 else []) + [(_s(truth + 1), "off")]
    if kind == "assignment":
        first = next(iter(truth))
        out = []
        if truth[first] != 0:
            out.append((", ".join(f"{k}={_s(-v if k == first else v)}" for k, v in truth.items()), "sign"))
        out.append((", ".join(f"{k}={_s(v + 1 if k == first else v)}" for k, v in truth.items()), "off"))
        return out
    raise ValueError(kind)


def _is_truly_correct(problem: dict, check: dict, answer: str, truth) -> bool:
    """채점 엔진과 독립적인 정답 확인 (값 비교만, 표기 해석은 SymPy 파서로)."""
    symbols = problem_symbols(problem)
    kind = check["type"]
    if kind in ("factor", "equivalent"):
        return sp.expand(parse_math(answer.split("=")[-1], symbols) - truth) == 0
    if kind == "roots":
        vals = [parse_math(v, symbols) for v in re.sub(r"x=", "", answer).split(",")]
        return {sp.nsimplify(v) for v in vals} == {sp.nsimplify(t) for t in truth}
    if kind == "value":
        return sp.simplify(parse_math(answer, symbols) - truth) == 0
    if kind == "assignment":
        vals = dict(p.split("=") for p in answer.replace(" ", "").split(","))
        return vals.keys() == truth.keys() and all(sp.simplify(parse_math(vals[k]) - truth[k]) == 0 for k in truth)
    raise ValueError(kind)


def build_dataset(problems: dict) -> list[dict]:
    items = []
    for p in problems.values():
        for i, step in enumerate(p["steps"]):
            check = step["check"]
            truth = _canonical(p, check)
            base = {"problem_id": p["id"], "step": i, "unit": p.get("type", "")}
            for ans in _correct_variants(p, check, truth):
                items.append({**base, "answer": ans, "label_correct": True, "category": "correct",
                              "label_diagnosis": NONE})
            misconception_answers = []
            for m in check.get("misconceptions", []):
                misconception_answers.append(m["answer"])
                items.append({**base, "answer": m["answer"], "label_correct": False, "category": "misconception",
                              "label_diagnosis": m["diagnosis"]})
            for ans, cat in _wrong_variants(p, check, truth):
                # 부호를 바꿔도 정답이거나(대칭 해 등) 이미 오개념 답과 같은 경우는 제외한다.
                if ans in misconception_answers or _is_truly_correct(p, check, ans, truth):
                    continue
                items.append({**base, "answer": ans, "label_correct": False, "category": cat,
                              "label_diagnosis": SIGN if cat == "sign" else OTHER})
    for n, item in enumerate(items):
        item["id"] = n
    return items


# ---------------------------------------------------------------- 판정 조건

def candidates(check: dict) -> list[str]:
    seen = []
    for m in check.get("misconceptions", []):
        if m["diagnosis"] not in seen:
            seen.append(m["diagnosis"])
    return seen + [SIGN, OTHER]


def judge_sympy(problem: dict, check: dict, answer: str) -> dict:
    r = check_step(problem, check, answer)
    if r.correct:
        diagnosis = NONE
    elif r.diagnosis in candidates(check):
        diagnosis = r.diagnosis
    elif r.error is ErrorType.SIGN:
        diagnosis = SIGN
    else:
        diagnosis = OTHER
    return {"correct": r.correct, "diagnosis": diagnosis, "ungraded": r.severity is Severity.UNGRADED}


JUDGE_INSTRUCTIONS = """너는 중학교 수학 학생 답안을 채점하는 채점자다.
- 학생 답이 주어진 풀이 단계의 답으로 수학적으로 옳은지 판정한다.
- 표기 방식(순서, 단위, 소수와 분수, √ 기호, 곱셈 기호 생략)이 달라도 값이 같으면 정답이다.
- 틀렸다면 오류 원인 후보 중 가장 알맞은 것을 고른다. 정답이면 diagnosis는 "없음"이다."""


def judge_llm(problem: dict, check: dict, answer: str, goal: str, reference: str | None,
              model: str, effort: str, usage: llm.Usage) -> dict:
    options = candidates(check) + [NONE]
    schema = {
        "type": "object",
        "properties": {"correct": {"type": "boolean"}, "diagnosis": {"type": "string", "enum": options}},
        "required": ["correct", "diagnosis"],
        "additionalProperties": False,
    }
    lines = [f"문제: {problem['statement']}", f"풀이 단계: {goal}"]
    if reference is not None:
        lines.append(f"이 단계의 정답: {reference}")
    lines += [f"학생 답: {answer}", f"오류 원인 후보: {', '.join(options)}"]
    complete = llm.json_completer(model, JUDGE_INSTRUCTIONS, schema, "judgement", effort, usage)
    out = complete([{"role": "user", "content": "\n".join(lines)}])
    return {"correct": out["correct"], "diagnosis": out["diagnosis"], "ungraded": False}


# ---------------------------------------------------------------- 실행과 집계

def summarize(items: list[dict], preds: dict[int, dict]) -> dict:
    def rate(subset, fn):
        subset = [i for i in subset if i["id"] in preds]
        return round(sum(fn(i, preds[i["id"]]) for i in subset) / len(subset), 4) if subset else None

    wrong = [i for i in items if not i["label_correct"]]
    right = [i for i in items if i["label_correct"]]
    by_unit = defaultdict(list)
    for i in items:
        by_unit[i["unit"]].append(i)
    return {
        "n": len(preds),
        "accuracy": rate(items, lambda i, p: p["correct"] == i["label_correct"]),
        "false_accept_rate": rate(wrong, lambda i, p: p["correct"]),
        "false_reject_rate": rate(right, lambda i, p: not p["correct"]),
        "misconception_diagnosis_match": rate([i for i in wrong if i["category"] == "misconception"],
                                              lambda i, p: p["diagnosis"] == i["label_diagnosis"]),
        "sign_diagnosis_match": rate([i for i in wrong if i["category"] == "sign"],
                                     lambda i, p: p["diagnosis"] == SIGN),
        "ungraded": sum(p.get("ungraded", False) for p in preds.values()),
        "accuracy_by_unit": {u: rate(v, lambda i, p: p["correct"] == i["label_correct"]) for u, v in by_unit.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=["sympy", "llm", "llm_ref"])
    parser.add_argument("--model", default=os.getenv("TUTOR_MODEL"))
    parser.add_argument("--effort", default="low")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="파일럿용: 앞에서부터 n개만")
    args = parser.parse_args()

    problems = {k: p for k, p in load_problems().items() if k not in CASE_STUDY_IDS}
    items = build_dataset(problems)[: args.limit]
    print(f"평가 데이터 {len(items)}개 (정답 {sum(i['label_correct'] for i in items)}, "
          f"오답 {sum(not i['label_correct'] for i in items)}: {dict(Counter(i['category'] for i in items))})")

    results = {"dataset": items, "conditions": {}}
    for cond in args.conditions:
        usage, started = llm.Usage(), time.time()

        def run(item):
            p = problems[item["problem_id"]]
            step = p["steps"][item["step"]]
            t0 = time.time()
            try:
                if cond == "sympy":
                    pred = judge_sympy(p, step["check"], item["answer"])
                else:
                    ref = _correct_variants(p, step["check"], _canonical(p, step["check"]))[0] if cond == "llm_ref" else None
                    pred = judge_llm(p, step["check"], item["answer"], step["goal"], ref, args.model, args.effort, usage)
            except Exception as e:  # API 오류도 결과로 남긴다
                pred = {"correct": None, "diagnosis": None, "error": repr(e)}
            pred["latency_sec"] = round(time.time() - t0, 3)
            return item["id"], pred

        workers = 1 if cond == "sympy" else args.workers
        with ThreadPoolExecutor(max_workers=workers) as pool:
            preds = dict(pool.map(run, items))
        ok = {k: v for k, v in preds.items() if v.get("correct") is not None}
        summary = summarize(items, ok)
        summary.update({"api_errors": len(preds) - len(ok), "usage": usage.to_dict(),
                        "elapsed_sec": round(time.time() - started, 1),
                        "mean_latency_sec": round(sum(v["latency_sec"] for v in ok.values()) / max(len(ok), 1), 3)})
        results["conditions"][cond] = {"summary": summary, "predictions": preds}
        print(f"\n[{cond}] " + json.dumps(summary, ensure_ascii=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"exp2_{'-'.join(args.conditions)}_{datetime.now():%Y%m%d-%H%M}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
