# 학습자 모델 기반 생성형 AI 수학 튜터

학생이 왜 틀렸는지(오개념)를 짚어, 튜터 없이도 다음 문제를 풀 수 있게 하는 대화형 수학 튜터.
범위: 중학교 인수분해·이차방정식(활용 포함), 미정계수, 연립일차방정식, 통계, 도형(계산형) — 22개 개념, 38문제.

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
python -m experiments.exp4_student_pilot --model gpt-5.4 --effort none --units algebra   # 실험 4 예비 시험: 가상 학생의 오개념 유지
python -m experiments.exp4_tutoring --first-turn answer   # 실험 4 (주 실험): 오개념 교정, --misconceptions로 오개념 지정
python -m experiments.case_stuck              # 사례: 막힌 학생 (활용 문제 W1)
python -m experiments.exp1_solver_verification --model gpt-5.4-nano --effort none --method sympy   # 실험 1: 풀이 검증 (--method self: LLM 자기 검증)
python -m experiments.exp2_judgement          # 실험 2: 판정·진단 정확도 (SymPy vs LLM)
python -m experiments.exp3_leakage            # 실험 3: 정답 요구 시 정답 유출률
```

실험 결과는 `experiments/results/`에 JSON으로 저장된다.

## 노트북 ↔ 데스크톱 작업 규칙

- 작업 시작 전: `git pull`
- 작업 종료 전: `git add` → `git commit` → `git push`
- `.env`(API 키)와 `docs/`(연구 노트·보고서)는 git에 올라가지 않으므로 PC마다 따로 관리한다.
