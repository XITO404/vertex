"""
XGBoost (13 features) vs ResNet (13 features) 비교 + 앙상블
선행 조건: select_features.py → resnet_optuna.py 완료 후 실행
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
PLOT_DIR = os.path.join(ROOT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

sys.path.insert(0, ROOT_DIR)
from data_preprocessing import preprocess

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── ResNet 구조 (resnet_optuna.py 와 동일해야 로드 가능) ─────────────────────

class ResBlock(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
    def forward(self, x):
        return x + self.block(x)


class TabResNet(nn.Module):
    def __init__(self, in_features, hidden_size, num_blocks, dropout):
        super().__init__()
        self.stem = nn.Linear(in_features, hidden_size)
        self.blocks = nn.Sequential(*[ResBlock(hidden_size, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
    def forward(self, x):
        return self.head(self.blocks(self.stem(x))).squeeze(1)


# ── 공통 평가 ────────────────────────────────────────────────────────────────

def metrics(y_true, y_pred, name=""):
    rmse_log = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2_log   = float(r2_score(y_true, y_pred))
    rmse_h   = float(np.sqrt(mean_squared_error(10**y_true, 10**y_pred)))
    r2_h     = float(r2_score(10**y_true, 10**y_pred))
    if name:
        print(f"\n  [{name}]")
        print(f"    log10  RMSE: {rmse_log:.4f}  R2: {r2_log:.4f}")
        print(f"    hours  RMSE: {rmse_h:.1f}  R2: {r2_h:.4f}")
    return {"rmse_log": rmse_log, "r2_log": r2_log, "rmse_h": rmse_h, "r2_h": r2_h}


# ── XGBoost 13-feature 재학습 ────────────────────────────────────────────────

def train_xgb13(X_train, y_train, X_test, y_test):
    print("XGBoost (13 features) 학습 중...")
    model = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    model.save_model(os.path.join(BASE_DIR, "xgb_13feat.json"))
    print(f"저장: {os.path.join(BASE_DIR, 'xgb_13feat.json')}")
    return model


# ── ResNet 로드 ───────────────────────────────────────────────────────────────

def load_resnet(selected):
    ckpt_path = os.path.join(BASE_DIR, "resnet_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} 없음. resnet_optuna.py 먼저 실행하세요.")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    params = ckpt["params"]
    model = TabResNet(
        in_features=len(selected),
        hidden_size=params["hidden_size"],
        num_blocks=params["num_blocks"],
        dropout=params["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"ResNet 로드 완료: hidden={params['hidden_size']}, blocks={params['num_blocks']}")
    return model


@torch.no_grad()
def predict_resnet(model, X):
    t = torch.tensor(X.values if hasattr(X, "values") else X, dtype=torch.float32).to(DEVICE)
    return model(t).cpu().numpy()


# ── 앙상블 ───────────────────────────────────────────────────────────────────

def stacking_ensemble(xgb_train, resnet_train, y_train, xgb_test, resnet_test):
    """Ridge 메타 학습기로 스태킹 앙상블."""
    S_train = np.column_stack([xgb_train, resnet_train])
    S_test  = np.column_stack([xgb_test,  resnet_test])
    meta = Ridge(alpha=1.0)
    meta.fit(S_train, y_train)
    preds = meta.predict(S_test)
    print(f"\n  메타 학습기 가중치  XGBoost: {meta.coef_[0]:.3f}, ResNet: {meta.coef_[1]:.3f}")
    return preds, meta


def simple_ensemble(xgb_pred, resnet_pred, weights=(0.5, 0.5)):
    return weights[0] * xgb_pred + weights[1] * resnet_pred


# ── 시각화 ───────────────────────────────────────────────────────────────────

def plot_comparison(y_test, results, save_path=None):
    names = list(results.keys())
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 12),
                             gridspec_kw={"height_ratios": [3, 2]})

    colors = ["#1565C0", "#B71C1C", "#1B5E20", "#4A148C"]

    # 상단: Pred vs Actual
    for i, name in enumerate(names):
        ax = axes[0, i]
        preds = results[name]["preds"]
        ax.scatter(y_test, preds, alpha=0.45, s=18,
                   color=colors[i], edgecolors="k", linewidths=0.2)
        lo = min(y_test.min(), preds.min()) - 0.1
        hi = max(y_test.max(), preds.max()) + 0.1
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel(r"실제값 $\log_{10}$(수명)", fontsize=10)
        ax.set_ylabel(r"예측값 $\log_{10}$(수명)", fontsize=10)
        m = results[name]
        ax.set_title(f"{name}\nRMSE={m['rmse_log']:.4f}  R2={m['r2_log']:.4f}", fontsize=11)
        ax.grid(True, alpha=0.3)

    # 하단: 지표 막대 비교
    metric_pairs = [("RMSE (log10)", "rmse_log"), ("R2 (log10)", "r2_log"),
                    ("RMSE (hours)", "rmse_h"),   ("R2 (hours)", "r2_h")]

    for j, (label, key) in enumerate(metric_pairs):
        if j >= n:
            break
        ax = axes[1, j]
        vals = [results[nm][key] for nm in names]
        bars = ax.bar(names, vals, color=colors[:n], edgecolor="k", linewidth=0.4)
        for bar, v in zip(bars, vals):
            fmt = f"{v:.4f}" if abs(v) < 100 else f"{v:.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_title(label, fontsize=11)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    # 남은 하단 칸 처리
    if n < len(metric_pairs):
        for j in range(n, len(axes[1])):
            axes[1, j].set_visible(False)

    fig.suptitle("모델 비교: XGBoost vs ResNet vs 앙상블", fontsize=14, y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"저장: {save_path}")
    plt.close(fig)


def plot_summary_bar(results, save_path=None):
    """R2와 RMSE(log10)를 나란히 정리한 요약 그래프."""
    names = list(results.keys())
    r2s   = [results[n]["r2_log"]   for n in names]
    rmses = [results[n]["rmse_log"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#1565C0", "#B71C1C", "#1B5E20", "#4A148C"]

    for ax, vals, title, fmt in zip(
        axes,
        [r2s, rmses],
        ["R2 (log10, 높을수록 좋음)", "RMSE (log10, 낮을수록 좋음)"],
        [".4f", ".4f"],
    ):
        bars = ax.bar(names, vals, color=colors[:len(names)], edgecolor="k", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:{fmt}}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("최종 모델 성능 요약", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"저장: {save_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  XGBoost (13) vs ResNet (13) 비교 + 앙상블")
    print("=" * 60)

    # 선택된 피처 로드
    feat_path = os.path.join(BASE_DIR, "selected_features.json")
    with open(feat_path, encoding="utf-8") as f:
        selected = json.load(f)
    print(f"사용 피처 ({len(selected)}개): {selected}\n")

    # 전처리
    data = preprocess(save=False)
    X_train = data["X_train"][selected]
    X_test  = data["X_test"][selected]
    y_train = data["y_train"].values
    y_test  = data["y_test"].values

    # ── 1. XGBoost 13-feature 재학습 ─────────────────────────────────────────
    print("\n[1/4] XGBoost 재학습 (13 features)")
    print("-" * 40)
    xgb = train_xgb13(X_train, y_train, X_test, y_test)
    xgb_train_pred = xgb.predict(X_train)
    xgb_test_pred  = xgb.predict(X_test)

    # ── 2. ResNet 로드 ────────────────────────────────────────────────────────
    print("\n[2/4] ResNet 로드")
    print("-" * 40)
    resnet = load_resnet(selected)
    resnet_train_pred = predict_resnet(resnet, X_train)
    resnet_test_pred  = predict_resnet(resnet, X_test)

    # ── 3. 평가 ───────────────────────────────────────────────────────────────
    print("\n[3/4] 개별 모델 평가")
    print("-" * 40)
    m_xgb    = metrics(y_test, xgb_test_pred,    "XGBoost (13)")
    m_resnet = metrics(y_test, resnet_test_pred,  "ResNet  (13)")

    # ── 4. 앙상블 ────────────────────────────────────────────────────────────
    print("\n[4/4] 앙상블")
    print("-" * 40)

    # 4-a. 단순 평균
    avg_pred = simple_ensemble(xgb_test_pred, resnet_test_pred, weights=(0.5, 0.5))
    m_avg = metrics(y_test, avg_pred, "Ensemble (평균)")

    # 4-b. 스태킹 (Ridge 메타 학습기)
    stack_pred, meta = stacking_ensemble(
        xgb_train_pred, resnet_train_pred, y_train,
        xgb_test_pred,  resnet_test_pred,
    )
    m_stack = metrics(y_test, stack_pred, "Ensemble (스태킹)")

    # ── 결과 정리 ─────────────────────────────────────────────────────────────
    results = {
        "XGBoost":          {**m_xgb,    "preds": xgb_test_pred},
        "ResNet":           {**m_resnet, "preds": resnet_test_pred},
        "앙상블(평균)":     {**m_avg,    "preds": avg_pred},
        "앙상블(스태킹)":   {**m_stack,  "preds": stack_pred},
    }

    print("\n" + "=" * 60)
    print("  [최종 비교 결과]")
    print("=" * 60)
    summary = pd.DataFrame({
        name: {k: v for k, v in r.items() if k != "preds"}
        for name, r in results.items()
    }).T
    summary.columns = ["RMSE(log10)", "R2(log10)", "RMSE(hours)", "R2(hours)"]
    summary["RMSE rank"] = summary["RMSE(log10)"].rank().astype(int)
    summary["R2 rank"]   = summary["R2(log10)"].rank(ascending=False).astype(int)
    print(summary.to_string())

    best = summary["R2(log10)"].idxmax()
    print(f"\n>>> 최고 성능 모델: {best}")
    print(f"    R2   (log10) = {summary.loc[best, 'R2(log10)']:.4f}")
    print(f"    RMSE (log10) = {summary.loc[best, 'RMSE(log10)']:.4f}")

    # ── 시각화 ────────────────────────────────────────────────────────────────
    plot_comparison(
        y_test, results,
        save_path=os.path.join(PLOT_DIR, "ensemble_comparison.png"),
    )
    plot_summary_bar(
        results,
        save_path=os.path.join(PLOT_DIR, "ensemble_summary.png"),
    )

    # 앙상블 모델 저장
    joblib.dump({
        "xgb": xgb,
        "resnet_params": torch.load(os.path.join(BASE_DIR, "resnet_best.pt"), map_location="cpu")["params"],
        "meta": meta,
        "selected_features": selected,
    }, os.path.join(BASE_DIR, "ensemble.pkl"))
    print(f"\n앙상블 저장: {os.path.join(BASE_DIR, 'ensemble.pkl')}")

    return results, summary


if __name__ == "__main__":
    main()
