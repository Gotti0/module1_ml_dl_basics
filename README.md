# 모듈 1. ML/DL 모델링 기초와 실험설계

## 소개

이 저장소는 **멀티캠퍼스 AI 엔지니어 트랙 — 모듈 1. ML/DL 모델링 기초와 실험설계**의 실습 자료를 모아둔 공간입니다.

- **이론 교안:** [모듈 1 이론 교안](노션 링크)에서 학습할 수 있습니다.
- **실습 코드:** Day 1~5 실습 노트북 코드가 포함되어 있습니다.
- **실습 방법:** 모든 노트북에는 실행 결과가 포함되어 있습니다. 결과를 먼저 살펴본 뒤, 직접 코드를 실행하고 비교해 보세요.

이론을 이해하는 것에서 끝나지 않고, **직접 실험하고 결과를 해석하며 모델의 동작을 이해하는 것**이 이번 모듈의 핵심 목표입니다.


## 환경 세팅
1. uv 설치 (한 번만)
   - Mac:     curl -LsSf https://astral.sh/uv/install.sh | sh
            brew install libomp
   - Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   설치 후 터미널 재시작

2. 레포 받기
   - git clone https://github.com/아이디/module1_ml_dl_basics.git
   - cd module1_ml_dl_basics

3. 환경 만들기
   - uv sync

4. VS Code로 이 폴더를 열고, 노트북 오른쪽 위 커널 선택에서
   - "module1-ml-dl-basics (Python 3.11)" 선택