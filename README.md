# AI 기반 크립 수명 예측 및 합금 설계 시스템

> 고온·고압 환경 핵심 소재의 크리프 파단 수명을 예측하고, 수명을 최대화하는 최적 합금 조성을 AI로 도출하는 웹 애플리케이션

---

## 📁 프로젝트 구조

```
vertex/
├── data/
│   ├── taka.xlsx              # 원본 데이터 (2066행 × 31열)
│   └── preprocessor.pkl       # 저장된 StandardScaler + 피처 메타정보
├── models/
│   ├── train.py               # XGBoost 모델 학습/평가
│   └── compare_models.py       # MLP, RF 모델 학습/ 평가
├── data_preprocessing.py      # 데이터 전처리 파이프라인
└── README.md                  # 본 문서
```

---

### 1. 데이터 전처리 파이프라인 (`data_preprocessing.py`)
- 원본 데이터(taka.xlsx) 로드: **2066행 × 31열**
- 물리적 불가능값 제거 (음수 온도, 0 이하 수명 등만 제거 / 조성 0은 유지)
- 타겟 변수 로그 변환: `log10(lifetime)`
- 냉각 속도 원핫 인코딩: `Cooling1/2/3` (0~3) → 12개 더미 컬럼
- Train/Test 분할: 80/20 (1652 / 414)
- StandardScaler 적용 (학습 데이터 기준 fit)
- `preprocessor.pkl` 저장 (scaler + feature_names + 메타정보)
- **최종 피처 수: 39개**

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
| ML/Data | Pandas, Scikit-learn, XGBoost |
| 최적화 | DEAP (유전 알고리즘) |
| 백엔드 | FastAPI |
| 프론트엔드 | Streamlit |
| 시각화 | Matplotlib |
