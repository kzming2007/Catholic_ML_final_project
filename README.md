<div align="center">
  
# 🤖 UR3 CobotOps Machine Learning Project
**산업용 협동로봇(UR3) 내부 센서 데이터를 활용한 가상 센싱 및 고장 예측 모델링**<br>
*Catholic University Machine Learning Final Project (2026)*

[![Python](https://img.shields.io/badge/Python-3.12-tested-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
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
├── dataset/            # UCI UR3 CobotOps 원본 데이터셋
├── notebooks/          # Task별 모델링 및 시각화 주피터 노트북
│   ├── 01_EDA.ipynb
│   ├── 02_Regression/
│   ├── 03_Classification/
│   └── 04_Clustering/
├── references/         # 논문, 레퍼런스 문서
├── reports/            # 최종 보고서 및 발표 자료 (PDF)
├── research/           # cycle-aware 재검증 및 시계열 모델 비교
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
* **수업 프로젝트 결과**: Row random split에서 `Random Forest` + `SMOTE` 파이프라인이 **F1-Macro Score 0.80, ROC-AUC 0.94**를 기록함.
* **인사이트**: 선형 모델보다 비선형 앙상블이 유리했고, 전류 계열 변수가 feature importance 상위에 반복적으로 나타남. 이는 예측에 유용한 지표라는 뜻이며 고장의 인과적 원인으로 해석하지 않음.

### 3. Clustering: "로봇은 몇 가지 상태로 움직이는가?"
* **목표**: 정답 라벨 없이 로봇의 운영 상태를 비지도 학습으로 군집화
* **수업 프로젝트 결과**: 전류 변수만을 활용한 `K-Means (K=4)`가 Silhouette 0.3814를 기록했고, 특정 군집에서 Protective Stop 비율이 14.34%로 나타남.
* **인사이트**: 전류 공간에 서로 다른 운영 regime이 존재할 가능성을 확인함. 군집 명칭은 사후 해석이고 실제 상태 정답이 아니므로, 이 결과만으로 선제적 고장 감지 성능이 입증된 것은 아님.

<br>

## 📊 Dataset Description
* **출처**: [UCI Machine Learning Repository - UR3 CobotOps Dataset](https://archive.ics.uci.edu/dataset/963/ur3+cobotops), DOI `10.24432/C5J891`, 데이터 라이선스 `CC BY 4.0`
* **특징**: 
  * RTDE 인터페이스는 125 Hz로 동작하지만, 공개 CSV의 양수 Timestamp 간격 중앙값은 약 1.005초이므로 저장 데이터를 125 Hz 균일 시계열로 간주하지 않음
  * 7,409행 × 24열 (관절 J0\~J5의 Current, Speed, Temperature 등)
  * 다양한 하중(1\~3kg) 및 그리퍼 파지력(80\~120N) 조건에서의 픽앤플레이스 시나리오 기록

## 🔬 Research Extension

후속 연구에서는 비연속적으로 재등장하는 cycle ID를 `cycle_run`으로 분리하고, cycle 경계를 넘지 않는 시계열 window와 9개 acquisition block held-out 평가를 적용했습니다. 동일한 10-step×19-sensor 입력에서 통계 feature 기반 `SMOTE + Random Forest`, 1D CNN, 정상-only LSTM Autoencoder를 비교했습니다.

Random Forest의 event cycle recall은 `System_Failure` 0.9770, `ProtectiveStop` 0.9516, `GripLost` 0.9231이었고 정상 cycle 오경보율은 0.0286-0.0435였습니다. 1D CNN은 recall은 유사했지만 오경보율이 0.1913-0.2454로 증가했습니다. LSTM Autoencoder는 q95에서 event recall이 0-0.0645에 그쳐 현재 데이터에서는 Random Forest보다 불리했습니다.

이 결과는 이상이 이미 포함된 구간의 탐지이며 조기 고장 예측을 뜻하지 않습니다. First positive 이전만 사용한 별도 실험에서는 `GripLost`의 약한 사전 신호 가능성만 확인됐습니다.

기존 프로젝트와 연구 확장의 차이, 얻은 결과, 한계는 [연구 종합 문서](research/2026-08-23_project_to_research_synthesis.md)에 정리했습니다. 재현 명령과 현재 상태는 [연구 README](research/README.md), 동일 센서 비교 결과는 [최종 비교 결과](research/outputs/12_matched_lstm_autoencoder_comparison.md)에서 확인할 수 있습니다.

<br>

## 💻 Quick Start

```bash
# 1. 저장소 클론
git clone https://github.com/kzming2007/UR3_Cobot_ML.git
cd UR3_Cobot_ML

# 2. 연구 스크립트용 검증 환경 설치
python -m pip install -r research/requirements.txt

# 3. 기존 노트북 실행 시 Jupyter와 시각화 패키지 추가 설치
python -m pip install jupyter matplotlib seaborn scipy statsmodels
jupyter notebook notebooks/01_basic_EDA.ipynb
```

<br>

## 👥 Contributors
| 이름 | 소속 |
|---|---|
| **김택명** | 가톨릭대학교 인공지능학과 |
| **남궁도현** | 가톨릭대학교 인공지능학과 |
| **유현성** | 가톨릭대학교 인공지능학과 |

<br>

---
*If you find this project helpful, please consider giving it a ⭐!*
