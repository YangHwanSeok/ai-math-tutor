# 학습자 모델 기반 생성형 AI 수학 튜터

학생의 개념 이해 상태에 따라 질문·힌트 수준을 조절하고, 생성형 AI의 풀이를 SymPy로 검증하는 대화형 수학 튜터.
1차 범위: 인수분해 → 이차방정식 단원.

## 폴더 구조

```
mathtutor/            # 튜터 본체 (Python 패키지)
tests/                # pytest 테스트
docs/research-notes/  # 연구 노트 (날짜별)
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

## 노트북 ↔ 데스크톱 작업 규칙

- 작업 시작 전: `git pull`
- 작업 종료 전: 연구 노트 갱신 → `git add` → `git commit` → `git push`
- `.env`(API 키)는 git에 올라가지 않으므로 PC마다 따로 만든다.
