# 학습자 모델 기반 생성형 AI 수학 튜터

학생이 왜 틀렸는지(오개념)를 짚어, 튜터 없이도 다음 문제를 풀 수 있게 하는 대화형 수학 튜터.
범위: 중학교 인수분해·이차방정식(활용 포함), 미정계수, 연립일차방정식, 통계, 도형(계산형), 고등학교 미적분(다항함수·합성함수), 확률 — 31개 개념, 문제 은행 38문제.
오개념 모델: 5개 단원(대수·통계·도형·미적분·확률) 22개. 오개념을 SymPy로 계산되는 버그 규칙으로 표현하고, 학생 답이 예측 오답과 같으면 그 오개념으로 진단한다.

설계 원칙: 정오 판정·오개념 진단은 SymPy, 도움의 양(힌트 레벨)은 학습자 모델, LLM은 발화 해석·생성만 담당한다.
오개념 라이브러리와 일치하는 답은 단순 실수가 아닌 개념 오류로 다룬다. 튜터 LLM에는 정답을 주지 않는다.

## 폴더 구조

```
mathtutor/            # 튜터 본체 (Python 패키지)
data/                 # 개념 그래프, 문제 은행
experiments/          # 실험 스크립트와 결과
tests/                # pytest 테스트
```

## 새 PC에서 시작하기

```bash
git clone <저장소 URL>
cd ai-math-tutor
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # 그 후 .env 에 API 키 입력
python -m pytest
```

## 실행

```bash
python -m mathtutor.cli E1                    # 대화형 튜터 (CLI), 문제 ID 지정
python -m experiments.exp4_student_pilot --model gpt-5.4 --effort none --units algebra   # 실험 4 예비 시험 (--units algebra statistics geometry calculus probability, --only로 오개념 지정)
python -m experiments.exp4_tutoring --first-turn answer   # 실험 4 (주 실험): 오개념 교정, --misconceptions로 오개념 지정
python -m experiments.exp4_tutoring --conditions bridge --reuse-pre <결과 파일>   # 기존 실행의 사전 풀이로 조건 추가 (Bridge 등)
python -m experiments.exp4_tutoring --retry-errors <결과 파일>   # API 오류로 끊긴 대화만 다시 실행
python -m experiments.analyze_exp4 --answer <파일 ...> --work <파일 ...>   # 실험 4 결과 합산·짝지은 비교
python -m experiments.exp5_diagnosis          # 실험 5: 오개념 진단 비교 (본 작품 vs 제약 기반 vs LLM + 목록 vs LLM 자유 진단), --only로 확장 세트
python -m experiments.case_stuck              # 사례: 막힌 학생 (활용 문제 W1)
python -m experiments.exp1_solver_verification --model gpt-5.4-nano --effort none --method sympy   # 실험 1: 풀이 검증 (--method self: LLM 자기 검증)
python -m experiments.exp2_judgement          # 실험 2: 판정·진단 정확도 (SymPy vs LLM)
python -m experiments.exp3_leakage            # 실험 3: 정답 요구 시 정답 유출률
```

실험 결과는 `experiments/results/`에 JSON으로 저장된다.
