"""
Baseline Model Training: XGBoost for Creep Rupture Life Prediction.
Trains on preprocessed data, evaluates with RMSE / R², and produces
a Predicted-vs-Actual scatter plot.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 경로 설정 ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = BASE_DIR
PLOT_DIR = os.path.join(ROOT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# data_preprocessing 모듈을 import 할 수 있도록 경로 추가
sys.path.insert(0, ROOT_DIR)
from data_preprocessing import preprocess


# ── XGBoost 학습 ─────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_test, y_test):
    """XGBoost 회귀 모델 학습 및 평가."""

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    return model


def evaluate(model, X_test, y_test):
    """RMSE, R² 계산 (log10 스케일 + 원래 시간 스케일)."""
    y_pred = model.predict(X_test)

    # log10 스케일 지표
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_log = r2_score(y_test, y_pred)

    # 원래 시간 스케일로 역변환
    y_test_hours = 10 ** y_test
    y_pred_hours = 10 ** y_pred

    rmse_hours = np.sqrt(mean_squared_error(y_test_hours, y_pred_hours))
    r2_hours = r2_score(y_test_hours, y_pred_hours)

    print("\n" + "=" * 60)
    print("  모델 평가 결과")
    print("=" * 60)
    print(f"  [log10 스케일]  RMSE = {rmse_log:.4f},  R2 = {r2_log:.4f}")
    print(f"  [시간 스케일]    RMSE = {rmse_hours:.1f} 시간,  R2 = {r2_hours:.4f}")
    print("=" * 60)

    return y_pred, {"rmse_log": rmse_log, "r2_log": r2_log,
                    "rmse_hours": rmse_hours, "r2_hours": r2_hours}


def plot_pred_vs_actual(y_test, y_pred, metrics, save_path=None):
    """예측값 vs 실제값 산점도 (log10 스케일)."""
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(y_test, y_pred, alpha=0.5, s=20, edgecolors="k", linewidths=0.3)

    # 대각선 (완벽 예측선)
    lims = [
        min(y_test.min(), y_pred.min()) - 0.1,
        max(y_test.max(), y_pred.max()) + 0.1,
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="완벽 예측선")

    ax.set_xlabel(r"실제값  $\log_{10}$(수명/시간)", fontsize=13)
    ax.set_ylabel(r"예측값  $\log_{10}$(수명/시간)", fontsize=13)
    ax.set_title(
        f"XGBoost 베이스라인 — 예측 vs 실제\n"
        f"RMSE={metrics['rmse_log']:.4f}    R²={metrics['r2_log']:.4f}",
        fontsize=14,
    )
    ax.legend(fontsize=12)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"\n그래프 저장: {save_path}")

    plt.show()


def plot_feature_importance(model, feature_names, top_n=15, save_path=None):
    """상위 피처 중요도 수평 막대 그래프."""
    importance = model.feature_importances_
    idx = np.argsort(importance)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_names[i] for i in idx],
        importance[idx],
        color="steelblue",
        edgecolor="k",
        linewidth=0.5,
    )
    ax.set_xlabel("피처 중요도 (gain)", fontsize=13)
    ax.set_title(f"XGBoost 상위 {top_n} 피처 중요도", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"그래프 저장: {save_path}")

    plt.show()


# ── 메인 실행 ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  XGBoost 베이스라인 모델 학습")
    print("=" * 60)

    # 1. 전처리
    data = preprocess(save=True)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    # 2. 학습
    print("\n모델 학습 시작...\n")
    model = train_xgboost(X_train, y_train, X_test, y_test)

    # 3. 평가
    y_pred, metrics = evaluate(model, X_test, y_test)

    # 4. 모델 저장
    model_path = os.path.join(MODEL_DIR, "xgb_baseline.json")
    model.save_model(model_path)
    print(f"\n모델 저장: {model_path}")

    # 5. 시각화
    plot_pred_vs_actual(
        y_test, y_pred, metrics,
        save_path=os.path.join(PLOT_DIR, "pred_vs_actual.png"),
    )
    plot_feature_importance(
        model, feature_names, top_n=15,
        save_path=os.path.join(PLOT_DIR, "feature_importance.png"),
    )

    return model, metrics


if __name__ == "__main__":
    main()
