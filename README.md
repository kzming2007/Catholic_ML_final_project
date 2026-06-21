<div align="center">
  
# 🤖 UR3 CobotOps Machine Learning Project
**산업용 협동로봇(UR3) 내부 센서 데이터를 활용한 가상 센싱 및 고장 예측 모델링**<br>
*Catholic University Machine Learning Final Project (2025)*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 📌 Project Overview
본 프로젝트는 스마트 팩토리 환경에서 구동되는 **UR3 협동로봇**의 관절 센서 로그(전류, 속도, 온도)를 분석하여, 별도의 고가 외부 토크 센서나 비전 카메라 없이 로봇의 상태를 진단하고 예측하는 **소프트웨어 기반 가상 센싱(Virtual Sensing)**의 가능성을 검증합니다.

데이터의 극심한 클래스 불균형(정상 93:고장 7)과 비선형적 물리 특성이라는 한계를 극복하기 위해, 회귀(Regression) → 분류(Classification) → 군집화(Clustering)로 이어지는 3단계의 유기적인 머신러닝 패러다임 전환을 수행했습니다.

<br>

## 📂 Repository Structure
```text
UR3_Cobot_ML/
├── dataset/            # UCI UR3 CobotOps 원본 및 전처리된 데이터셋
├── notebooks/          # Task별 모델링 및 시각화 주피터 노트북
│   ├── 01_EDA.ipynb
│   ├── 02_Regression/
│   ├── 03_Classification/
│   └── 04_Clustering/
├── references/         # 논문, 레퍼런스 문서 및 API 가이드
├── reports/            # 최종 보고서 및 발표 자료 (PDF)
├── .gitignore
└── README.md
```

<br>

## 🚀 Key Tasks & Findings

본 프로젝트는 단순한 알고리즘 성능 경쟁을 넘어, 로봇 동역학(Dynamics) 도메인 지식과 머신러닝을 융합하는 것에 집중했습니다.

### 1. Regression: "그리퍼 전류를 예측할 수 있는가?"
* **목표**: 6개 관절의 센서 데이터를 통해 말단 그리퍼의 작동 전류(Tool Current) 예측
* **결과**: `Random Forest` 모델이 최고 성능(R² 0.58) 달성.
* **인사이트**: 그리퍼 전류는 대기/전이/파지의 3단계 삼봉(Tri-modal) 분포를 띰. 연속형 예측(회귀) 모델은 존재하지 않는 중간 전류값을 예측하는 근본적 한계가 존재함을 증명하며 **분류 문제로의 전환 당위성**을 확보함.

### 2. Classification: "로봇 고장을 예측할 수 있는가?"
* **목표**: 보호 정지(Protective Stop) 및 파지 상실(Grip Lost) 에러 발생 유무 이진 분류
* **결과**: `Random Forest` + `SMOTE` 파이프라인으로 **F1-Macro Score 0.80, 오탐지(FP) 91% 감소** (LR 대비).
* **인사이트**: 선형 모델(LR)의 한계를 비선형 앙상블로 극복. 피처 중요도 분석 결과, 온도나 속도가 아닌 **'전류(Current) 변수'가 고장 예측의 핵심 신호**임을 밝혀냄.

### 3. Clustering: "로봇은 몇 가지 상태로 움직이는가?"
* **목표**: 정답 라벨 없이 로봇의 운영 상태를 비지도 학습으로 군집화
* **결과**: 전류 변수만을 활용한 `K-Means (K=4)` 모델이 로봇의 4가지 물리적 상태를 성공적으로 도출.
* **인사이트**: 분류 태스크의 발견을 독립 검증함. 특히, 도출된 **'과부하 상태(Cluster 2)'에 실제 에러의 14.3%가 밀집**되어 있음을 규명하여, 비지도 학습 기반의 선제적 고장 감지 시스템의 실효성을 입증함.

<br>

## 📊 Dataset Description
* **출처**: [UCI Machine Learning Repository - UR3 CobotOps Dataset](https://archive.ics.uci.edu/dataset/963/ur3+cobotops)
* **특징**: 
  * 125Hz 실시간 샘플링
  * 7,409행 × 24열 (관절 J0~J5의 Current, Speed, Temperature 등)
  * 다양한 하중(1~3kg) 및 그리퍼 파지력(80~120N) 조건에서의 픽앤플레이스 시나리오 기록

<br>

## 💻 Quick Start

```bash
# 1. 저장소 클론
git clone https://github.com/kzming2007/UR3_Cobot_ML.git
cd UR3_Cobot_ML

# 2. 필수 라이브러리 설치
pip install -r requirements.txt

# 3. 노트북 실행
jupyter notebook notebooks/01_EDA.ipynb
```
*(참고: `requirements.txt`가 없다면 scikit-learn, pandas, numpy, seaborn, imbalanced-learn 등이 필요합니다.)*

<br>

## 👥 Contributors
| 이름 | 소속 | 역할 및 기여 |
|---|---|---|
| **김택명** | 가톨릭대학교 인공지능학과 | 프로젝트 리딩, 머신러닝 모델 구축 및 최적화 |
| **남궁도현** | 가톨릭대학교 인공지능학과 | 데이터 전처리, 탐색적 데이터 분석(EDA), 문서화 |
| **유현성** | 가톨릭대학교 인공지능학과 | 도메인 지식 분석, 시각화 및 모델 성능 검증 |

<br>

---
*If you find this project helpful, please consider giving it a ⭐!*
