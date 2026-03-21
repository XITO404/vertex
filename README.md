# AI 기반 크립 수명 예측 및 합금 설계 시스템

> 고온·고압 환경 핵심 소재의 크리프(Creep) 파단 수명을 예측하고, 수명을 최대화하는 최적 합금 조성을 AI로 도출하는 웹 애플리케이션

---

## 📁 프로젝트 구조

```
vertex/
├── data/
│   ├── taka.xlsx              # 원본 데이터 (2066행 × 31열)
│   ├── preprocessor.pkl       # 저장된 StandardScaler + 피처 메타정보
│   └── correlation_heatmap.png # 전처리 결과 변수 간 상관관계 히트맵
├── documents/
│   └── 회의록.md               # 팀 프로젝트 진행 기록
├── models/
│   ├── train.py               # XGBoost 모델 학습/평가
│   └── compare_models.py       # MLP, RF 모델 학습/ 평가
├── data_preprocessing.py      # 데이터 전처리 및 피처 엔지니어링
├── streamlit_app.py           # Streamlit 기반 웹 애플리케이션 실행 파일
├── requirements.txt           # 프로젝트 라이브러리 의존성 목록
└── README.md                  # 본 문서
```

---

### 1. 데이터 전처리 및 피처 엔지니어링 (`data_preprocessing.py`)
- 원본 데이터(taka.xlsx) 로드: **2066행 × 31열**
- 데이터 정제: 결측치 처리(합금 성분 NaN → 0) 및 물리적 무결성 검사 (음수 온도 등 필터링)
- 이상치 분석: 전체 데이터의 약 14% 이상치 탐지 (응력 변수 중심 특이치 확인)
- 물리 기반 파생 변수: 소재 도메인 지식을 반영한 Severity Index (가혹도 지수) 3종 추가<bn>
: $N/T/A\_severity$: Hollomon-Jaffe 파라미터를 응용한 온도-시간 비선형 관계 수치화
- 냉각 방식 최적화: `Cooling1/2/3` (0~3) → 고정형 원핫 인코딩 적용 (12개 컬럼)
- 데이터 직렬화: 학습된 `StandardScaler`와 메타 정보를 `preprocessor.pkl`로 저장
- **최종 피처 수: 42개** (기존 31개 대비 11개 확장)

### 2. XGBoost 베이스라인 모델 (`models/train.py`)
- XGBoost 회귀 모델 학습 (500 estimators, max_depth=6, lr=0.05)
- **모델 성능:**

| 스케일 | RMSE | R² |
|--------|------|----|
| log10 | 0.2793 | **0.9168** |
| 시간(hours) | 8,741.4 | 0.7533 |

- 모델 저장: `xgb_baseline.json`
- 시각화: 예측 vs 실제 산점도, 피처 중요도 그래프

---


## 🛠 기술 스택

| 구분 | 기술 |
|------|------|
| 언어 | Python 3.13 |
| ML/Data | Pandas, NumPy, Scikit-learn, XGBoost, Seaborn, Joblib |
| 최적화 | DEAP (유전 알고리즘) |
| 백엔드 | FastAPI |
| 프론트엔드 | Streamlit |
| 시각화 | Matplotlib |
