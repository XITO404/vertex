"""
Data Preprocessing Pipeline for Creep Rupture Life Prediction.
Loads taka.xlsx, applies transformations per CLAUDE.md rules, and saves
the preprocessor object and processed data splits.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PATH = os.path.join(DATA_DIR, "taka.xlsx")

# ── Column definitions ──────────────────────────────────────────────────
TARGET = "lifetime"

COOLING_COLS = ["Cooling1", "Cooling2", "Cooling3"]
COOLING_LABELS = {0: "Furnace", 1: "Air", 2: "Oil", 3: "Water"}

COMPOSITION_COLS = [
    "C", "Si", "Mn", "P", "S", "Cr", "Mo", "W",
    "Ni", "Cu", "V", "Nb", "N", "Al", "B", "Co", "Ta", "O",
]
CONDITION_COLS = ["stress", "temp"]
HEAT_TREATMENT_COLS = [
    "Ntemp", "Ntime", "Ttemp", "Ttime", "Atemp", "Atime",
]
EXTRA_COLS = ["Rh"]

NUMERIC_COLS = CONDITION_COLS + COMPOSITION_COLS + HEAT_TREATMENT_COLS + EXTRA_COLS


# ── Helper functions ─────────────────────────────────────────────────────
def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw Excel file."""
    df = pd.read_excel(path)
    print(f"Loaded {path}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove physically impossible values only.
    - Negative lifetime, stress, or temperature
    - Zero or negative lifetime (log transform needs > 0)
    NOTE: 0 in composition columns is intentional (element not added).
    """
    n_before = len(df)
    mask = (
        (df[TARGET] > 0)
        & (df["stress"] > 0)
        & (df["temp"] > 0)
        & (df["Ntemp"] >= 0)
        & (df["Ttemp"] >= 0)
        & (df["Atemp"] >= 0)
    )
    df = df[mask].reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed:
        print(f"Removed {n_removed} rows with physically impossible values")
    else:
        print("No invalid rows found")
    return df


def log_transform_target(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log10 to the target variable (lifetime)."""
    df = df.copy()
    df["log_lifetime"] = np.log10(df[TARGET])
    print(
        f"Log10(lifetime) range: [{df['log_lifetime'].min():.3f}, "
        f"{df['log_lifetime'].max():.3f}]"
    )
    return df


def one_hot_encode_cooling(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode cooling rate columns (categorical: 0-3)."""
    df = df.copy()
    for col in COOLING_COLS:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
        # Ensure all categories exist
        for val, label in COOLING_LABELS.items():
            expected_col = f"{col}_{val}"
            if expected_col not in dummies.columns:
                dummies[expected_col] = 0
        # Sort columns for consistency
        dummies = dummies[[f"{col}_{v}" for v in sorted(COOLING_LABELS.keys())]]
        df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=COOLING_COLS)
    print(f"One-hot encoded {COOLING_COLS} -> {len(COOLING_LABELS) * len(COOLING_COLS)} columns")
    return df


def build_preprocessor(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the numeric features of the training set."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    print(f"StandardScaler fitted on {X_train.shape[1]} features")
    return scaler


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns after one-hot encoding."""
    exclude = {TARGET, "log_lifetime"}
    return [c for c in df.columns if c not in exclude]


# ── Main preprocessing pipeline ─────────────────────────────────────────
def preprocess(
    test_size: float = 0.2,
    random_state: int = 42,
    save: bool = True,
) -> dict:
    """
    Full preprocessing pipeline:
      1. Load raw data
      2. Remove physically impossible values
      3. Log-transform target
      4. One-hot encode cooling rate columns
      5. Train/test split
      6. Fit StandardScaler on train set, transform both
      7. Save preprocessor and processed data
    Returns dict with keys: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    print("=" * 60)
    print("  Data Preprocessing Pipeline")
    print("=" * 60)

    # 1. Load
    df = load_raw_data()

    # 2. Remove invalid
    df = remove_invalid_rows(df)

    # 3. Log-transform target
    df = log_transform_target(df)

    # 4. One-hot encode cooling
    df = one_hot_encode_cooling(df)

    # 5. Feature / target split
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["log_lifetime"]

    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print(f"Samples: {len(X)}")

    # 6. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 7. Scale
    scaler = build_preprocessor(X_train)
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=feature_cols, index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index,
    )

    # 8. Save artifacts
    if save:
        preprocessor_path = os.path.join(DATA_DIR, "preprocessor.pkl")
        joblib.dump(
            {
                "scaler": scaler,
                "feature_names": feature_cols,
                "target": "log_lifetime",
                "cooling_labels": COOLING_LABELS,
            },
            preprocessor_path,
        )
        print(f"\nPreprocessor saved to {preprocessor_path}")

    print("=" * 60)
    print("  Preprocessing complete")
    print("=" * 60)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_cols,
    }


if __name__ == "__main__":
    result = preprocess()
    print(f"\nX_train shape: {result['X_train'].shape}")
    print(f"X_test  shape: {result['X_test'].shape}")
    print(f"y_train range: [{result['y_train'].min():.3f}, {result['y_train'].max():.3f}]")
    print(f"y_test  range: [{result['y_test'].min():.3f}, {result['y_test'].max():.3f}]")
