"""실험 4 결과 합산·짝지은 비교.

같은 사전 풀이(p1, rep)에서 갈라진 조건끼리 짝지어 확인 문제 정답률의 차이를 구하고,
시행 단위 부트스트랩(4,000회)으로 95% 신뢰구간을 낸다. Bridge처럼 나중에 --reuse-pre로 추가한 조건 파일도 합친다.

실행: python -m experiments.analyze_exp4 --answer <파일 ...> --work <파일 ...>
"""
import argparse
import json
import random
from pathlib import Path

from experiments.student_sim import UNITS

CONDITIONS = ["retry_only", "chatbot", "prompt_tutor", "prompt_tutor_solution", "bridge", "proposed", "proposed_no_diag"]
UNIT_OF = {mid: unit for unit, mids in UNITS.items() for mid in mids}


def load(files: list[str]) -> dict:
    """{(p1, rep): {condition: 확인 문제 정답률}}과 대화 목록을 만든다."""
    trials: dict = {}
    dialogs = []
    for f in files:
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        for d in data["dialogs"]:
            if "error" in d:
                continue
            score = sum(p["label"] == "correct" for p in d["post"]) / len(d["post"])
            trials.setdefault((d["p1"], d["rep"]), {})[d["condition"]] = score
            dialogs.append(d)
    return {"trials": trials, "dialogs": dialogs}


def _ci(values: list[float], seed: int = 0, n: int = 4000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return [round(means[int(0.025 * n)] * 100, 1), round(means[int(0.975 * n) - 1] * 100, 1)]


def _pct(values: list[float]) -> float | None:
    return round(sum(values) / len(values) * 100, 1) if values else None


def analyze(data: dict, units: list[str] | None = None) -> dict:
    trials = {k: v for k, v in data["trials"].items()
              if units is None or UNIT_OF[k[0].rsplit("-", 1)[0]] in units}
    out = {"trials": len(trials), "rate": {}, "n": {}, "diff_vs_proposed": {}}
    for cond in CONDITIONS:
        vals = [t[cond] for t in trials.values() if cond in t]
        if vals:
            out["rate"][cond], out["n"][cond] = _pct(vals), len(vals)
    for cond in CONDITIONS:
        if cond == "proposed":
            continue
        diffs = [t["proposed"] - t[cond] for t in trials.values() if "proposed" in t and cond in t]
        if diffs:
            out["diff_vs_proposed"][cond] = {"diff": _pct(diffs), "ci95": _ci(diffs), "n": len(diffs)}
    leaks = {}
    for cond in CONDITIONS:
        ds = [d for d in data["dialogs"] if d["condition"] == cond and d.get("tutor_leaked") is not None
              and (units is None or UNIT_OF[d["misconception"]] in units)]
        if ds:
            leaks[cond] = _pct([d["tutor_leaked"] for d in ds])
    out["leak_rate"] = leaks
    by_mid = {}
    for mid in UNIT_OF:
        rows = {cond: _pct([t[cond] for k, t in trials.items() if k[0].rsplit("-", 1)[0] == mid and cond in t])
                for cond in CONDITIONS}
        if any(v is not None for v in rows.values()):
            by_mid[mid] = rows
    out["by_misconception"] = by_mid
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer", nargs="+", required=True)
    parser.add_argument("--work", nargs="+", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = {}
    for setting, files in (("answer", args.answer), ("work", args.work)):
        data = load(files)
        result[setting] = {"all": analyze(data)}
        for unit in UNITS:
            r = analyze(data, [unit])
            if r["trials"]:
                result[setting][unit] = r
        result[setting]["statistics+geometry"] = analyze(data, ["statistics", "geometry"])
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    for setting in result:
        for scope, r in result[setting].items():
            print(f"[{setting} / {scope}] 시행 {r['trials']}  " +
                  "  ".join(f"{c}={v}" for c, v in r["rate"].items()))
            print("   제안-: " + "  ".join(f"{c} {v['diff']:+} {v['ci95']}" for c, v in r["diff_vs_proposed"].items()))


if __name__ == "__main__":
    main()
