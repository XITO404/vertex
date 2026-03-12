"""
모델 비교 스크립트: XGBoost vs Random Forest vs MLP
기존 xgb_baseline.json을 건드리지 않고, 동일 데이터로 3개 모델을 학습·평가·비교합니다.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 경로 설정 ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
PLOT_DIR = os.path.join(ROOT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

sys.path.insert(0, ROOT_DIR)
from data_preprocessing import preprocess
from models.train import train_xgboost


# ── 모델 학습 함수 ───────────────────────────────────────────────────────
def train_random_forest(X_train, y_train):
    """Random Forest 회귀 모델 학습"""
    print("\nRandom Forest 모델 학습 중...")
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    return rf_model


def train_mlp(X_train, y_train):
    """
    MLP (다층 퍼셉트론) 회귀 모델 학습.
    NOTE: preprocess()가 이미 StandardScaler를 적용하므로
          여기서는 MLPRegressor만 사용합니다 (이중 스케일링 방지).
    """
    print("\nMLP 모델 학습 중...")
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.01,
        max_iter=500,
        random_state=42,
        early_stopping=True,
    )
    mlp_model.fit(X_train, y_train)
    return mlp_model


# ── 공통 평가 함수 ───────────────────────────────────────────────────────
def evaluate_model(model_name, model, X_test, y_test):
    """모델 평가: log10 및 원래 시간 스케일 지표 반환"""
    y_pred = model.predict(X_test)

    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_log = r2_score(y_test, y_pred)

    y_test_hours = 10 ** y_test
    y_pred_hours = 10 ** y_pred

    rmse_hours = np.sqrt(mean_squared_error(y_test_hours, y_pred_hours))
    r2_hours = r2_score(y_test_hours, y_pred_hours)

    print(f"\n[{model_name} 평가 결과]")
    print("-" * 40)
    print(f"  [log10 스케일] RMSE = {rmse_log:.4f}, R² = {r2_log:.4f}")
    print(f"  [시간 스케일]   RMSE = {rmse_hours:.1f} 시간, R² = {r2_hours:.4f}")
    print("-" * 40)

    return {
        "y_pred": y_pred,
        "rmse_log": rmse_log,
        "r2_log": r2_log,
        "rmse_hours": rmse_hours,
        "r2_hours": r2_hours,
    }


# ── 비교 시각화 ──────────────────────────────────────────────────────────
def plot_comparison(y_test, results, save_path=None):
    """3모델 Pred vs Actual 산점도 + Metrics 막대 그래프"""
    names = list(results.keys())
    n = len(names)

    fig, axes = plt.subplots(2, n, figsize=(6 * n, 11),
                             gridspec_kw={"height_ratios": [3, 2]})

    # ── 상단: Pred vs Actual 산점도 ──────────────────────────────────────
    for i, name in enumerate(names):
        ax = axes[0, i]
        y_pred = results[name]["y_pred"]

        ax.scatter(y_test, y_pred, alpha=0.5, s=20, edgecolors="k", linewidths=0.3)
        lims = [
            min(y_test.min(), y_pred.min()) - 0.1,
            max(y_test.max(), y_pred.max()) + 0.1,
        ]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="완벽 예측선")
        ax.set_xlabel(r"실제값 $\log_{10}$(수명)", fontsize=11)
        ax.set_ylabel(r"예측값 $\log_{10}$(수명)", fontsize=11)
        ax.set_title(
            f"{name}\n"
            f"RMSE={results[name]['rmse_log']:.4f}  R²={results[name]['r2_log']:.4f}",
            fontsize=12,
        )
        ax.legend(fontsize=9)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # ── 하단: Metrics 막대 비교 ──────────────────────────────────────────
    metrics_labels = ["RMSE (log10)", "R² (log10)", "RMSE (시간)", "R² (시간)"]
    metric_keys = ["rmse_log", "r2_log", "rmse_hours", "r2_hours"]

    for j, (label, key) in enumerate(zip(metrics_labels, metric_keys)):
        ax = axes[1, j] if j < n else None
        if ax is None:
            break
        vals = [results[name][key] for name in names]
        colors = ["#4C72B0", "#55A868", "#C44E52"]
        bars = ax.bar(names, vals, color=colors[:n], edgecolor="k", linewidth=0.5)

        # 값 레이블
        for bar, v in zip(bars, vals):
            fmt = f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title(label, fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("모델 성능 비교: XGBoost vs Random Forest vs MLP", fontsize=15, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n비교 그래프 저장: {save_path}")

    plt.close(fig)


# ── 메인 실행 ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  모델 비교 실험: XGBoost vs Random Forest vs MLP")
    print("=" * 60)

    # 1. 전처리 (동일 데이터 사용)
    data = preprocess(save=False)  # 기존 preprocessor.pkl 덮어쓰지 않음
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    results = {}

    # 2. XGBoost 학습 (기존 xgb_baseline.json은 건드리지 않음)
    print("\n" + "=" * 60)
    print("  [1/3] XGBoost 학습")
    print("=" * 60)
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
    results["XGBoost"] = evaluate_model("XGBoost", xgb_model, X_test, y_test)

    # 3. Random Forest 학습
    print("\n" + "=" * 60)
    print("  [2/3] Random Forest 학습")
    print("=" * 60)
    rf_model = train_random_forest(X_train, y_train)
    results["Random Forest"] = evaluate_model("Random Forest", rf_model, X_test, y_test)

    # 4. MLP 학습
    print("\n" + "=" * 60)
    print("  [3/3] MLP (Neural Network) 학습")
    print("=" * 60)
    mlp_model = train_mlp(X_train, y_train)
    results["MLP"] = evaluate_model("MLP", mlp_model, X_test, y_test)

    # 5. 비교 테이블 출력
    print("\n\n" + "=" * 60)
    print("  ★ 최종 모델 비교 결과 ★")
    print("=" * 60)
    summary = pd.DataFrame({
        name: {k: v for k, v in vals.items() if k != "y_pred"}
        for name, vals in results.items()
    }).T
    summary.columns = ["RMSE (log₁₀)", "R² (log₁₀)", "RMSE (시간)", "R² (시간)"]

    # 순위 매기기
    summary["RMSE 순위"] = summary["RMSE (log₁₀)"].rank().astype(int)
    summary["R² 순위"] = summary["R² (log₁₀)"].rank(ascending=False).astype(int)

    print(summary.to_string())

    best_model = summary["R² (log₁₀)"].idxmax()
    print(f"\n>>> 최고 성능 모델 (R² 기준): {best_model}")
    print(f"    R² (log₁₀) = {summary.loc[best_model, 'R² (log₁₀)']:.4f}")
    print(f"    RMSE (log₁₀) = {summary.loc[best_model, 'RMSE (log₁₀)']:.4f}")

    # 6. 비교 시각화
    plot_comparison(
        y_test, results,
        save_path=os.path.join(PLOT_DIR, "model_comparison.png"),
    )

    return results, summary


if __name__ == "__main__":
    main()
