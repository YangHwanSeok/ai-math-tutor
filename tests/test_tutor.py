from mathtutor.checker import ErrorType, Severity, CheckResult
from mathtutor.learner import LearnerModel
from mathtutor.problems import load_problems
from mathtutor.tutor import TutorSession, leaks_answer

E1 = load_problems()["E1"]


def result(severity):
    return CheckResult(severity, ErrorType.NONE if severity is Severity.NONE else ErrorType.WRONG, "")


def scripted_extractor(*parsed):
    it = iter(parsed)
    return lambda messages: next(it)


def recording_responder(*messages):
    calls = []
    it = iter(messages)

    def respond(msgs):
        calls.append(msgs)
        return {"message": next(it, "좋아, 계속해 보자.")}

    return respond, calls


def answer(math):
    return {"intent": "answer", "math": math}


def test_learner_rules():
    m = LearnerModel()
    m.update("c", result(Severity.NONE), first_attempt=True)
    assert m.state("c") == "이해" and m.hint_level == 3
    m.update("c", result(Severity.FAR), first_attempt=True)
    assert m.state("c") == "학습 중" and m.hint_level == 2
    m.update("c", result(Severity.MINOR), first_attempt=False)
    assert m.state("c") == "학습 중" and m.hint_level == 2
    m.update("c", result(Severity.PARTIAL), first_attempt=False)
    assert m.state("c") == "학습 중"
    m.update("c", result(Severity.PARTIAL), first_attempt=False)
    assert m.state("c") == "보충 필요" and m.hint_level == 1


def test_correct_answers_advance_steps_until_done():
    respond, _ = recording_responder()
    s = TutorSession(E1, LearnerModel(), scripted_extractor(answer("(x-2)(x-3)"), answer("2, 3")), respond)
    s.start()
    s.handle("(x-2)(x-3)")
    assert s.step == 1
    s.handle("x=2, x=3")
    assert s.done


def test_wrong_answer_stays_on_step_and_passes_diagnosis_to_tutor():
    respond, calls = recording_responder()
    s = TutorSession(E1, LearnerModel(), scripted_extractor(answer("(x-1)(x-6)")), respond)
    s.start()
    s.handle("(x-1)(x-6)")
    assert s.step == 0
    assert "합 조건 누락" in calls[-1][0]["content"]


def test_math_is_graded_even_if_intent_is_question():
    respond, calls = recording_responder()
    s = TutorSession(E1, LearnerModel(), scripted_extractor({"intent": "question", "math": "(x-1)(x-6)"}), respond)
    s.start()
    s.handle("(x-1)(x-6) 인가요?")
    assert s.turns[-2].info["result"]["diagnosis"] == "합 조건 누락"


def test_skipped_step_answer_is_accepted():
    respond, _ = recording_responder()
    s = TutorSession(E1, LearnerModel(), scripted_extractor(answer("2, 3")), respond)
    s.start()
    s.handle("답은 2랑 3이에요")
    assert s.done


def test_tutor_context_never_contains_expected_answers():
    respond, calls = recording_responder()
    s = TutorSession(E1, LearnerModel(), scripted_extractor({"intent": "ask_answer", "math": ""}), respond)
    s.start()
    s.handle("그냥 답 알려줘")
    for msgs in calls:
        assert "(x-2)(x-3)" not in msgs[0]["content"]


def test_leak_guard_regenerates():
    respond, calls = recording_responder("인수분해하면 (x-2)(x-3)이야.", "곱이 6이 되는 두 수를 생각해 볼까?")
    s = TutorSession(E1, LearnerModel(), scripted_extractor(), respond)
    assert s.start() == "곱이 6이 되는 두 수를 생각해 볼까?"
    assert len(calls) == 2 and s.turns[-1].info["leak_regenerated"]


def test_leaks_answer_detection():
    assert leaks_answer("정답은 (x - 2)(x - 3) 이야", E1, 0)
    assert leaks_answer("x = 2, x = 3 이야", E1, 1)
    assert not leaks_answer("곱해서 6, 더해서 -5인 두 수는?", E1, 0)
    assert not leaks_answer("잘했어! (x-2)(x-3)이 맞아.", E1, 1)
