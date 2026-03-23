# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-based creep rupture life prediction system for high-temperature/high-pressure alloy materials. Predicts creep rupture lifetime and can be used to derive optimal alloy compositions via ML models.

## Commands

### Run the Streamlit web app
```bash
streamlit run streamlit_app.py
```

### Train the XGBoost baseline model (saves `models/xgb_baseline.json` and `data/preprocessor.pkl`)
```bash
python models/train.py
```

### Compare XGBoost vs Random Forest vs MLP
```bash
python models/compare_models.py
```

### Run the preprocessing pipeline standalone
```bash
python data_preprocessing.py
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Architecture

### Data Flow
1. **Raw data**: `data/taka.xlsx` — 2066 rows × 31 columns (lifetime, stress, temperature, 18 alloy compositions in wt%, heat treatment conditions, 3 cooling rates)
2. **Preprocessing** (`data_preprocessing.py`): removes physically impossible values, applies `log10(lifetime)` target transform, one-hot encodes 3 cooling columns (0–3 each → 12 dummy columns), 80/20 train/test split, fits `StandardScaler` on train set → **39 features total**
3. **Preprocessor artifact**: `data/preprocessor.pkl` — stores `{scaler, feature_names, target, cooling_labels}`
4. **Model training**: `models/train.py` trains XGBoost (500 estimators, max_depth=6, lr=0.05); `models/compare_models.py` compares XGBoost, Random Forest, and MLP on the same data without overwriting `xgb_baseline.json`
5. **Streamlit app** (`streamlit_app.py`): calls `preprocess(save=False)` and trains the model in-process on first load (`@st.cache_resource`), then exposes a UI for entering stress, temperature, alloy composition (wt%), heat treatment conditions, and cooling rate to get a predicted lifetime

### Key Design Decisions
- Target is always modeled in `log10(hours)` space; predictions are exponentiated for display
- Composition columns with value `0` are valid (element not added); only negative values are removed
- The Streamlit app re-trains the model from scratch on startup rather than loading a saved model file — this keeps the app self-contained with no dependency on `xgb_baseline.json`
- `compare_models.py` imports `train_xgboost` from `models/train.py` and uses `preprocess(save=False)` to avoid modifying artifacts
- Plots are saved to `plots/` directory (auto-created); Korean labels use `Malgun Gothic` font

### Feature Columns (39 total after preprocessing)
- `stress`, `temp` (test conditions)
- 18 composition columns: `C, Si, Mn, P, S, Cr, Mo, W, Ni, Cu, V, Nb, N, Al, B, Co, Ta, O`
- 6 heat treatment columns: `Ntemp, Ntime, Ttemp, Ttime, Atemp, Atime`
- `Rh` (Rhenium wt%)
- 12 one-hot cooling columns: `Cooling1_0/1/2/3`, `Cooling2_0/1/2/3`, `Cooling3_0/1/2/3`
