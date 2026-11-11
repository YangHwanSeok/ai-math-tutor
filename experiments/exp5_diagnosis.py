"""실험 5: 오개념 진단 정확도. 학생의 답 하나만 보고 무슨 오개념인지 알아내는가.

같은 오답 세트를 세 방법으로 진단한다.
- library   : 본 작품. 오개념 규칙으로 계산한 예측 오답과 학생 답을 SymPy로 비교 (BUGGY 계열)
- llm_catalog: 기존 LLM 오개념 탐지 방식 (Eedi 'Mining Misconceptions in Mathematics'와 같은 과제).
              문제, 정답, 학생 답과 오개념 목록을 주고 LLM이 해당 오개념을 고르거나 '없음'을 고른다.
- llm_free  : 목록 없이 LLM이 오류를 직접 진단한다 (Bridge의 '오류 유형 추론'처럼 LLM 튜터가 스스로 판단하는 경우).
              일관된 잘못된 규칙(오개념)인지 우연한 실수인지와 그 내용을 쓰게 하고, 내용이 맞는지는 LLM 채점자가 본다.

오답 세트 (정답 라벨은 오답을 만든 방법에서 정해진다)
- bug    : 라이브러리 오개념대로 계산한 답 (17개 오개념 × 문제 6개)
- slip   : 정답에서 계수 하나·해 하나를 바꾸거나 값을 1~2 틀린 우연한 실수 (문제마다 2개, 무작위).
            오개념 답과 우연히 겹쳐도 그대로 둔다(라이브러리 방식의 오탐을 재기 위함).
- unseen : 라이브러리에 없는 오개념대로 계산한 답 (단원마다 1개 × 문제 6개, 정답과 같아지는 문제는 제외)

실행: python -m experiments.exp5_diagnosis [--models gpt-5.4 gpt-5.4-mini] [--reps 3]
"""
import argparse
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import sympy as sp

from experiments.student_sim import MISCONCEPTIONS, UNITS, _poly, _terms, build_problem, correct_answer
from mathtutor import llm
from mathtutor.checker import _CHECKERS, check_step, parse_math, problem_symbols, x

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _fmt(expr) -> str:
    return str(expr).replace("**", "^")


# ---------- 라이브러리에 없는 오개념 (단원마다 1개) ----------
def _roots_of(text: str) -> list[sp.Expr]:
    left, right = text.split("=")
    return sp.solve(parse_math(left) - parse_math(right), x)


UNSEEN = {
    "algebra": {
        "source": "divide_by_x",
        "diagnosis": "이항할 때 부호를 바꾸지 않음 (x^2=7x → x^2+7x=0)",
        "answer": lambda text: ", ".join(_fmt(r) for r in sp.solve(
            parse_math(text.split("=")[0]) + parse_math(text.split("=")[1]), x)),
    },
    "statistics": {
        "source": "variance_no_divide",
        "diagnosis": "편차를 제곱하지 않고 절댓값의 평균을 구함 (분산과 평균편차 혼동)",
        "answer": lambda v: _fmt(sum(abs(a - sp.Rational(sum(v), len(v))) for a in v) / len(v)),
    },
    "geometry": {
        "source": "pythagoras_no_sqrt",
        "diagnosis": "빗변의 길이를 직각을 낀 두 변의 길이의 합으로 계산",
        "answer": lambda ab: _fmt(ab[0] + ab[1]),
    },
    "probability": {
        "source": "prob_at_least_one_add",
        "diagnosis": "'적어도 한 번'의 확률을 구할 때 여사건의 확률을 1에서 빼지 않음 (모두 일어나지 않을 확률을 답으로 씀)",
        "answer": lambda p: _fmt((1 - p[0]) ** p[1]),
    },
    "calculus": {
        "source": "power_rule_keep_exponent",
        "diagnosis": "x^n을 미분할 때 지수만 1 줄이고 n을 앞으로 내려 곱하지 않음 (ax^n → ax^(n-1))",
        "answer": lambda text: _fmt(sum(c * x ** (n - 1) for c, n in _terms(_poly(text)) if n > 0)),
    },
}


# ---------- 우연한 실수 ----------
def _slip(problem: dict, rng: random.Random) -> str:
    """정답의 한 군데만 조금 바꾼 답을 만든다."""
    check = problem["steps"][0]["check"]
    kind = check["type"]
    if kind == "value":
        v = parse_math(check["expected"])
        return _fmt(rng.choice([v + 1, v - 1, v + 2, v - 2]))
    if kind == "roots":
        while True:
            roots = [parse_math(r) for r in check["expected"]]
            i = rng.randrange(len(roots))
            roots[i] = rng.choice([roots[i] + 1, roots[i] - 1, -roots[i]])
            if len(set(roots)) == len(roots):  # 해 두 개가 같아지면 다시 만든다
                return ", ".join(_fmt(r) for r in roots)
    if kind == "factor":
        factors = list(sp.Mul.make_args(sp.factor(parse_math(check["expected"]))))
        linear = [i for i, f in enumerate(factors) if f.has(x) and sp.Poly(f, x).degree() == 1]
        i = rng.choice(linear)
        c = factors[i].subs(x, 0)
        factors[i] = factors[i] - c + rng.choice([c + 1, c - 1, -c])
        return "".join(f"({_fmt(f)})" for f in factors)
    # 식이 답인 경우(도함수, 부정적분): 항 하나의 계수를 바꾼다
    target = parse_math(check["target"].replace("**", "^"))
    terms = _terms(target)
    c, n = rng.choice(terms)
    new = target - c * x ** n + rng.choice([c + 1, c - 1, -c]) * x ** n
    return _fmt(sp.expand(new)) + (" + C" if kind == "antiderivative" else "")


def build_items(seed: int = 0, only: list[str] | None = None) -> list[dict]:
    """only를 주면 그 오개념들의 문항(과 그 단원의 목록 밖 오개념 문항)만 만든다. 재심 보강의 확장 세트용."""
    rng = random.Random(seed)
    items = []
    for unit, mids in UNITS.items():
        for mid in mids:
            if only and mid not in only:
                continue
            for i in range(len(MISCONCEPTIONS[mid]["problems"])):
                p = build_problem(mid, i)
                bug = p["steps"][0]["check"]["misconceptions"][0]["answer"]
                items.append({"unit": unit, "problem": p["id"], "kind": "bug", "truth": mid, "answer": _fmt(bug)})
                for _ in range(2):
                    items.append({"unit": unit, "problem": p["id"], "kind": "slip", "truth": "none",
                                  "answer": _slip(p, rng)})
        u = UNSEEN[unit]
        if only and u["source"] not in only:
            continue
        src = MISCONCEPTIONS[u["source"]]
        for i, text in enumerate(src["problems"]):
            items.append({"unit": unit, "problem": f"{u['source']}-{i}", "kind": "unseen", "truth": "none",
                          "unseen_diagnosis": u["diagnosis"], "answer": u["answer"](text)})
    for it in items:
        mid, i = it["problem"].rsplit("-", 1)
        it["_p"] = build_problem(mid, int(i))
    # 정답과 같아진 실수 답은 뺀다
    return [it for it in items if not check_step(it["_p"], it["_p"]["steps"][0]["check"], it["answer"]).correct]


# ---------- 방법 1: 오개념 라이브러리 ----------
LIBRARY_DIAGNOSES = {m["diagnosis"]: mid for mid, m in MISCONCEPTIONS.items()}


def diagnose_library(item: dict) -> dict:
    started = time.perf_counter()
    p = item["_p"]
    result = check_step(p, p["steps"][0]["check"], item["answer"])
    pred = LIBRARY_DIAGNOSES.get(result.diagnosis, "none") if result.misconception else "none"
    return {"pred": pred, "ms": round((time.perf_counter() - started) * 1000, 2)}


# ---------- 방법 1-b: 제약 기반 진단 (고전 ITS) ----------
def diagnose_constraint(item: dict) -> dict:
    """오개념 라이브러리 없이 채점 규칙(조건 위반)만으로 진단한다 (Ohlsson의 제약 기반 모델링에 해당).

    무엇이 틀렸는지(부호, 일부 누락, 어느 차수의 계수 등)는 알려 주지만 어떤 오개념인지는 이름 붙이지 못한다.
    '경미(MINOR)'로 판정되면 실수로 본 것으로 센다.
    """
    p = item["_p"]
    check = p["steps"][0]["check"]
    symbols = problem_symbols(p)
    try:
        result = _CHECKERS[check["type"]](p, check, item["answer"], symbols)
    except Exception:  # 해석할 수 없는 답
        return {"severity": "UNGRADED", "error": "UNPARSEABLE"}
    return {"severity": result.severity.name, "error": result.error.name, "diagnosis": result.diagnosis}


# ---------- 방법 2: LLM + 오개념 목록 ----------
CATALOG = "\n".join(f"- {mid}: {m['diagnosis']}" for mid, m in MISCONCEPTIONS.items())
CATALOG_PROMPT = f"""너는 수학 교사다. 학생의 오답을 보고, 아래 오개념 목록 중 학생이 가진 오개념을 고른다.
학생 답이 목록의 어느 오개념으로도 설명되지 않으면(단순 실수이거나 목록에 없는 오개념) none을 고른다.
reason에는 판단 근거를 한 문장으로 쓴다.

[오개념 목록]
{CATALOG}"""
CATALOG_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"},
                   "misconception": {"type": "string", "enum": [*MISCONCEPTIONS, "none"]}},
    "required": ["reason", "misconception"],
    "additionalProperties": False,
}

# ---------- 방법 3: LLM 자유 진단 ----------
FREE_PROMPT = """너는 수학 교사다. 학생의 오답을 보고 오류를 진단한다.
- kind: 학생이 일관되게 믿는 잘못된 규칙이 있어 보이면 misconception, 우연한 계산 실수로 보이면 slip.
- description: misconception이면 학생이 믿는 잘못된 규칙을 한 문장으로 쓴다. slip이면 어디서 실수했는지 쓴다."""
FREE_SCHEMA = {
    "type": "object",
    "properties": {"kind": {"type": "string", "enum": ["misconception", "slip"]}, "description": {"type": "string"}},
    "required": ["kind", "description"],
    "additionalProperties": False,
}
JUDGE_PROMPT = """두 진단이 같은 오개념(같은 잘못된 규칙)을 말하는지 판정한다. 표현이 달라도 같은 규칙이면 same=true.
[기준 진단]과 다른 규칙이거나 더 막연하면(예: '계산 실수', '부호 실수'만 말함) same=false."""
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"same": {"type": "boolean"}},
    "required": ["same"],
    "additionalProperties": False,
}


def _question(item: dict) -> str:
    p = item["_p"]
    return f"문제: {p['statement']}\n정답: {correct_answer(p)}\n학생 답: {item['answer']}"


def _timed(fn, messages):
    started = time.perf_counter()
    out = fn(messages)
    return out, round((time.perf_counter() - started) * 1000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gpt-5.4", "gpt-5.4-mini"])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--judge-model", default="gpt-5.4")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="+", default=None, help="이 오개념들의 문항만 (확장 세트)")
    parser.add_argument("--methods", nargs="+", default=["llm_catalog", "llm_free"])
    args = parser.parse_args()

    items = build_items(only=args.only)[: args.limit]
    usage = {m: llm.Usage() for m in args.models}
    judge_usage = llm.Usage()
    judge = llm.json_completer(args.judge_model, JUDGE_PROMPT, JUDGE_SCHEMA, "judge", "low", judge_usage)

    records = []
    for it in items:
        records.append({k: v for k, v in it.items() if k != "_p"} | {"library": diagnose_library(it),
                                                                      "constraint": diagnose_constraint(it)})

    def run(task):
        idx, model, method, rep = task
        it = items[idx]
        msg = [{"role": "user", "content": _question(it)}]
        try:
            if method == "llm_catalog":
                fn = llm.json_completer(model, CATALOG_PROMPT, CATALOG_SCHEMA, "diagnosis", args.effort, usage[model])
                out, ms = _timed(fn, msg)
                return task, {"pred": out["misconception"], "reason": out["reason"], "ms": ms}
            fn = llm.json_completer(model, FREE_PROMPT, FREE_SCHEMA, "diagnosis", args.effort, usage[model])
            out, ms = _timed(fn, msg)
            res = {"kind": out["kind"], "description": out["description"], "ms": ms}
            reference = (MISCONCEPTIONS[it["truth"]]["diagnosis"] if it["kind"] == "bug"
                         else it.get("unseen_diagnosis"))
            if reference and out["kind"] == "misconception":
                verdict = judge([{"role": "user",
                                  "content": f"[기준 진단] {reference}\n[비교 진단] {out['description']}"}])
                res["same"] = verdict["same"]
            return task, res
        except Exception as e:  # API 오류도 결과로 남긴다
            return task, {"error": repr(e)}

    tasks = [(i, m, method, r) for i in range(len(items)) for m in args.models
             for method in args.methods for r in range(args.reps)]
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for (i, model, method, rep), res in pool.map(run, tasks):
            records[i].setdefault(f"{method}:{model}", []).append(res)

    summary = summarize(records, args.models)
    meta = {"models": args.models, "only": args.only, "catalog_size": len(MISCONCEPTIONS), "reps": args.reps, "effort": args.effort, "judge_model": args.judge_model,
            "items": len(items), "items_by_kind": dict(Counter(it["kind"] for it in items)),
            "usage": {m: u.to_dict() for m, u in usage.items()} | {"judge": judge_usage.to_dict()},
            "elapsed_sec": round(time.time() - started, 1)}
    print(json.dumps({"meta": meta, "summary": summary}, ensure_ascii=False, indent=2))
    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = "_ext" if args.only else ""
    out = RESULTS_DIR / f"exp5_diagnosis{suffix}_{datetime.now():%Y%m%d-%H%M}.json"
    out.write_text(json.dumps({"meta": meta, "summary": summary, "records": records}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"저장: {out}")


def _rate(values) -> float | None:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else None


def _correct_catalog(rec: dict, pred: str) -> bool:
    return pred == rec["truth"]


def _correct_free(rec: dict, res: dict) -> bool:
    if rec["kind"] == "slip":
        return res["kind"] == "slip"
    return res["kind"] == "misconception" and res.get("same", False)


def constraint_summary(records: list[dict]) -> dict:
    """제약 기반 진단: 오개념 답을 '경미(실수 수준)'로 본 비율, 실수 답을 '경미'로 본 비율. 오개념 이름은 붙이지 못한다(식별 0%)."""
    out = {"bug_identified": 0.0}
    for kind in ("bug", "slip", "unseen"):
        rs = [r for r in records if r["kind"] == kind]
        out[f"{kind}_as_minor"] = _rate(r["constraint"]["severity"] == "MINOR" for r in rs)
    out["bug_as_minor_by_unit"] = {u: _rate(r["constraint"]["severity"] == "MINOR" for r in records
                                            if r["kind"] == "bug" and r["unit"] == u) for u in UNITS}
    return out


def summarize(records: list[dict], models: list[str]) -> dict:
    kinds = ["bug", "slip", "unseen"]
    summary = {"library": {}}
    for kind in kinds:
        rs = [r for r in records if r["kind"] == kind]
        summary["library"][kind] = _rate(_correct_catalog(r, r["library"]["pred"]) for r in rs)
    summary["library"]["slip_flagged_as_misconception"] = _rate(
        r["library"]["pred"] != "none" for r in records if r["kind"] == "slip")
    summary["library"]["ms_per_item"] = _rate(r["library"]["ms"] for r in records)
    summary["constraint"] = constraint_summary(records)

    for model in models:
        for method, correct in (("llm_catalog", lambda r, x: _correct_catalog(r, x["pred"])),
                                ("llm_free", _correct_free)):
            key = f"{method}:{model}"
            s = {}
            for kind in kinds:
                rs = [r for r in records if r["kind"] == kind]
                s[kind] = _rate(correct(r, x) for r in rs for x in r.get(key, []) if "error" not in x)
            ok = [x for r in records for x in r.get(key, []) if "error" not in x]
            if method == "llm_catalog":
                s["slip_flagged_as_misconception"] = _rate(
                    x["pred"] != "none" for r in records if r["kind"] == "slip" for x in r.get(key, []) if "error" not in x)
                s["unseen_forced_into_library"] = _rate(
                    x["pred"] != "none" for r in records if r["kind"] == "unseen" for x in r.get(key, [])
                    if "error" not in x)
                answers = [tuple(x.get("pred") for x in r.get(key, [])) for r in records]
            else:
                s["slip_flagged_as_misconception"] = _rate(
                    x["kind"] == "misconception" for r in records if r["kind"] == "slip" for x in r.get(key, [])
                    if "error" not in x)
                answers = [tuple(x.get("kind") for x in r.get(key, [])) for r in records]
            s["consistent_across_reps"] = _rate(len(set(a)) == 1 for a in answers if len(a) > 1)
            s["ms_per_item"] = _rate(x["ms"] for x in ok)
            s["errors"] = sum("error" in x for r in records for x in r.get(key, []))
            summary[key] = s
    # 단원별 bug 정확도
    summary["bug_by_unit"] = {
        unit: {"library": _rate(r["library"]["pred"] == r["truth"] for r in records if r["unit"] == unit and r["kind"] == "bug"),
               **{f"llm_catalog:{m}": _rate(x["pred"] == r["truth"] for r in records
                                            if r["unit"] == unit and r["kind"] == "bug"
                                            for x in r.get(f"llm_catalog:{m}", []) if "error" not in x)
                  for m in models}}
        for unit in UNITS}
    return summary


if __name__ == "__main__":
    main()
