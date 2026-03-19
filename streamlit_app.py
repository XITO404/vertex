import os
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

from data_preprocessing import preprocess


@st.cache_resource
def load_and_train():
    data = preprocess(save=False)
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
    model.fit(data["X_train"], data["y_train"])
    return model, data["feature_names"]


def main():
    st.set_page_config(page_title="Vertex — 크립 수명 예측", layout="wide")
    st.title("AI 기반 크립 수명 예측 (Baseline)")
    st.caption("현재 리포지토리의 전처리/학습 스크립트를 기반으로, 로컬에서 즉시 실행 가능한 Streamlit 웹앱입니다.")

    data_path = os.path.join(os.path.dirname(__file__), "data", "taka.xlsx")
    if not os.path.exists(data_path):
        st.error(f"`data/taka.xlsx`를 찾을 수 없습니다: {data_path}")
        st.stop()

    with st.spinner("데이터 전처리 및 XGBoost 학습 중... (최초 1회)"):
        model, feature_names = load_and_train()

    st.success("준비 완료")

    st.subheader("입력값")
    cols = st.columns(3)

    # 기본 입력(자주 쓰는 조건 변수)
    stress = cols[0].number_input("Rupture stress (MPa) = stress", min_value=0.0, value=150.0, step=1.0)
    temp = cols[1].number_input("Temperature (Kelvin) = temp", min_value=0.0, value=873.0, step=1.0)

    # 나머지 피처는 0으로 기본값 제공 (합금 조성 미입력 시 baseline 형태)
    row = {name: 0.0 for name in feature_names}
    if "stress" in row:
        row["stress"] = float(stress)
    if "temp" in row:
        row["temp"] = float(temp)

    # 대표 합금 원소 몇 개만 노출 (나머지는 0)
    st.markdown("**대표 조성(wt%)** (선택 입력, 나머지는 0 처리)")
    c2 = st.columns(6)
    for i, k in enumerate(["Cr", "Mo", "W", "Ni", "V", "Nb"]):
        if k in row:
            row[k] = float(c2[i].number_input(k, min_value=0.0, value=0.0, step=0.1))

    # 열처리 조건(있으면 노출)
    st.markdown("**열처리 조건** (선택 입력)")
    c3 = st.columns(6)
    for i, k in enumerate(["Ntemp", "Ntime", "Ttemp", "Ttime", "Atemp", "Atime"]):
        if k in row:
            row[k] = float(c3[i].number_input(k, min_value=0.0, value=0.0, step=1.0))

    # 원-핫 냉각 조건은 기본 Furnace(0)로 두고, 사용자가 바꾸면 반영
    st.markdown("**냉각 속도(원-핫)**")
    cooling = st.selectbox("Cooling (0=furnace, 1=air, 2=oil, 3=water)", options=[0, 1, 2, 3], index=0)
    for prefix in ["Cooling1_", "Cooling2_", "Cooling3_"]:
        for v in [0, 1, 2, 3]:
            key = f"{prefix}{v}"
            if key in row:
                row[key] = 1.0 if v == int(cooling) else 0.0

    X = pd.DataFrame([row], columns=feature_names)

    if st.button("수명 예측", type="primary"):
        y_log = float(model.predict(X)[0])
        y_hours = 10 ** y_log
        st.metric("예측 log10(수명/시간)", f"{y_log:.4f}")
        st.metric("예측 수명(시간)", f"{y_hours:,.1f}")


if __name__ == "__main__":
    main()

