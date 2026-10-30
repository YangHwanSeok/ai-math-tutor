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


def test_messages_after_done_are_not_graded():
    respond, calls = recording_responder()
    s = TutorSession(E1, LearnerModel(), scripted_extractor(answer("2, 3")), respond)
    s.start()
    s.handle("답은 2랑 3이에요")
    assert s.done
    s.handle("감사합니다!")  # 추출기를 더 부르지 않고, 단계 범위를 넘지 않아야 한다
    assert "모든 단계를 완료함" in calls[-1][0]["content"]


def test_wrong_answer_stays_on_step_and_passes_diagnosis_to_tutor():
    respond, calls = recording_responder()
    s = TutorSession(E1, LearnerModel(), scripted_extractor(answer("(x-1)(x-6)")), respond)
    s.start()
    s.handle("(x-1)(x-6)")
    assert s.step == 0
    assert "합 조건 누락" in calls[-1][0]["content"]


def test_misconception_is_flagged_to_tutor():
    respond, calls = recording_responder()
    u1 = load_problems()["U1"]
    s = TutorSession(u1, LearnerModel(), scripted_extractor(answer("x^2+(b+2)x+2b"), answer("a=6, b=3")), respond)
    s.start()
    s.handle("x^2+(b+2)x+2b")
    s.handle("a=6, b=3")
    assert "[오개념] 합·곱 조건 혼동" in calls[-1][0]["content"]
    assert "[오개념]" not in calls[-2][0]["content"]


def test_diagnosis_can_be_hidden_for_ablation():
    respond, calls = recording_responder()
    s = TutorSession(E1, LearnerModel(), scripted_extractor(answer("(x-1)(x-6)")), respond, show_diagnosis=False)
    s.start()
    s.handle("(x-1)(x-6)")
    context = calls[-1][0]["content"]
    assert "[채점 결과] 오답" in context and "합 조건 누락" not in context and "일차항" not in context
    assert s.learner.hint_level == 1  # 학습자 모델은 그대로 갱신된다


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


def test_answer_requests_are_counted_and_do_not_raise_help():
    respond, calls = recording_responder()
    ask = {"intent": "ask_answer", "math": ""}
    s = TutorSession(E1, LearnerModel(), scripted_extractor(ask, ask), respond)
    s.start()
    s.handle("답 알려줘")
    s.handle("답만 알려달라고")
    assert s.answer_requests == 2
    assert "[정답 요구] 2번째 요구" in calls[-1][0]["content"]
    assert s.learner.hint_level == 2


def test_leak_guard_can_be_disabled_for_ablation():
    respond, calls = recording_responder("인수분해하면 (x-2)(x-3)이야.")
    s = TutorSession(E1, LearnerModel(), scripted_extractor(), respond, leak_guard=False)
    assert s.start() == "인수분해하면 (x-2)(x-3)이야."
    assert len(calls) == 1


def test_leak_guard_regenerates():
    respond, calls = recording_responder("인수분해하면 (x-2)(x-3)이야.", "곱이 6이 되는 두 수를 생각해 볼까?")
    s = TutorSession(E1, LearnerModel(), scripted_extractor(), respond)
    assert s.start() == "곱이 6이 되는 두 수를 생각해 볼까?"
    assert len(calls) == 2 and s.turns[-1].info["leak_regenerated"]


def test_leaks_answer_detection_for_assignment():
    u1 = {"unknowns": ["a", "b"], "condition": {"type": "identity", "left": "x^2+ax+6", "right": "(x+2)(x+b)"},
          "steps": [{"check": {"type": "assignment"}}]}
    assert leaks_answer("정답은 a = 5, b = 3 이야", u1, 0)
    assert not leaks_answer("a의 값부터 구해 볼까? b=3은 맞아.", u1, 0)


def test_leaks_answer_detection():
    assert leaks_answer("정답은 (x - 2)(x - 3) 이야", E1, 0)
    assert leaks_answer("x = 2, x = 3 이야", E1, 1)
    assert not leaks_answer("곱해서 6, 더해서 -5인 두 수는?", E1, 0)
    assert not leaks_answer("잘했어! (x-2)(x-3)이 맞아.", E1, 1)
