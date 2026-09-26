"""대화형 튜터 CLI.

실행: python -m mathtutor.cli E1
대본 재생: python -m mathtutor.cli E1 --script experiments/scripts/e1_struggling.txt
"""
import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from mathtutor.learner import LearnerModel
from mathtutor.llm import Usage
from mathtutor.problems import load_problems
from mathtutor.tutor import TutorSession, llm_extractor, llm_responder

DIALOG_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results" / "dialogs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_id")
    parser.add_argument("--script", help="학생 발화를 한 줄씩 적은 파일 (없으면 직접 입력)")
    parser.add_argument("--learner", help="학습자 상태 JSON 경로 (있으면 불러오고, 끝나면 저장)")
    args = parser.parse_args()

    learner_path = Path(args.learner) if args.learner else None
    learner = LearnerModel.load(learner_path) if learner_path and learner_path.exists() else LearnerModel()
    usage = Usage()
    session = TutorSession(load_problems()[args.problem_id], learner, llm_extractor(usage), llm_responder(usage))

    print(f"튜터: {session.start()}")
    script = iter(Path(args.script).read_text(encoding="utf-8").splitlines()) if args.script else None
    while not session.done:
        if script is not None:
            text = next(script, None)
            if text is None:
                break
            print(f"학생: {text}")
        else:
            text = input("학생: ")
        if text.strip() in ("q", "quit", "종료"):
            break
        print(f"튜터: {session.handle(text)}")

    print(f"\n[학습자 상태] {learner.concept_states} / 힌트 레벨 {learner.hint_level}")
    if learner_path:
        learner.save(learner_path)

    DIALOG_DIR.mkdir(parents=True, exist_ok=True)
    name = Path(args.script).stem if args.script else "interactive"
    out = DIALOG_DIR / f"{args.problem_id}_{name}_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({
        "problem_id": args.problem_id, "completed": session.done,
        "learner": {"concept_states": learner.concept_states, "hint_level": learner.hint_level},
        "usage": asdict(usage), "turns": [asdict(t) for t in session.turns],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"대화 기록 저장: {out}")


if __name__ == "__main__":
    main()
