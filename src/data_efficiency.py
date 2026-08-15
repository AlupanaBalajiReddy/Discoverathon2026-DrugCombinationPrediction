import os
import gc
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

FEATURES_DIR = "features"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

FRACTIONS = [0.01, 0.05, 0.10, 0.25]

RANDOM_STATE = 42


# ============================================================
# PREPROCESSING
# Same logic as final training pipeline
# ============================================================

LEAKAGE_COLUMNS = [
    "SCORE",
    "PERCENTGROWTH",
    "PERCENTGROWTHNOTZ",
    "TESTVALUE",
    "CONTROLVALUE",
    "TZVALUE",
    "EXPECTEDGROWTH"
]

METADATA_COLUMNS = [
    "SCREENER",
    "STUDY",
    "TESTDATE",
    "PLATE",
    "PREFIX1",
    "PREFIX2",
    "drug_pair_id"
]


def prepare_data(df, feature_columns=None):

    y = pd.to_numeric(
        df["SCORE"],
        errors="coerce"
    )

    remove_columns = (
        LEAKAGE_COLUMNS
        + METADATA_COLUMNS
    )

    remove_columns = [
        col
        for col in remove_columns
        if col in df.columns
    ]

    X = df.drop(
        columns=remove_columns
    )

    categorical_columns = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    if categorical_columns:

        X = pd.get_dummies(
            X,
            columns=categorical_columns,
            drop_first=False,
            dtype="int8"
        )

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.fillna(0)

    float_columns = X.select_dtypes(
        include=["float64"]
    ).columns

    X[float_columns] = X[
        float_columns
    ].astype("float32")

    if feature_columns is not None:

        X = X.reindex(
            columns=feature_columns,
            fill_value=0
        )

    return X, y


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 70)
print("DATA-EFFICIENCY EXPERIMENT")
print("=" * 70)

train_path = os.path.join(
    FEATURES_DIR,
    "random",
    "train_features.csv"
)

test_path = os.path.join(
    FEATURES_DIR,
    "random",
    "test_features.csv"
)

print("\nLoading training data...")

train_df = pd.read_csv(
    train_path,
    low_memory=False
)

print(
    "Raw training shape:",
    train_df.shape
)


# ============================================================
# PREPARE FULL TRAINING DATA
# ============================================================

X_full, y_full = prepare_data(
    train_df
)

del train_df
gc.collect()

print(
    "Training matrix:",
    X_full.shape
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

test_df = pd.read_csv(
    test_path,
    low_memory=False
)

X_test, y_test = prepare_data(
    test_df,
    feature_columns=X_full.columns
)

del test_df
gc.collect()

print(
    "Test matrix:",
    X_test.shape
)


# ============================================================
# DATA-EFFICIENCY RESULTS
# ============================================================

results = []


# ============================================================
# TRAIN SMALLER DATASETS
# ============================================================

for fraction in FRACTIONS:

    print("\n")
    print("=" * 70)
    print(
        f"TRAINING WITH {fraction * 100:.0f}% DATA"
    )
    print("=" * 70)

    n_samples = int(
        len(X_full) * fraction
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    indices = rng.choice(
        len(X_full),
        size=n_samples,
        replace=False
    )

    X_subset = X_full.iloc[
        indices
    ]

    y_subset = y_full.iloc[
        indices
    ]

    print(
        "Training samples:",
        n_samples
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=30,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    print(
        "Training Random Forest..."
    )

    model.fit(
        X_subset,
        y_subset
    )

    print(
        "Training complete."
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print(
        "Generating predictions..."
    )

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    results.append({
        "Training_Fraction": fraction,
        "Training_Percentage": fraction * 100,
        "Training_Samples": n_samples,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    del X_subset
    del y_subset
    del predictions
    del model

    gc.collect()


# ============================================================
# ADD EXISTING 100% RESULT
# ============================================================

full_result = {
    "Training_Fraction": 1.0,
    "Training_Percentage": 100.0,
    "Training_Samples": len(X_full),
    "RMSE": 10.331846504295697,
    "MAE": 6.851900179697948,
    "R2": 0.3101510343016579
}

results.append(
    full_result
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "Training_Fraction"
)

output_path = os.path.join(
    RESULTS_DIR,
    "data_efficiency.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\n")
print("=" * 70)
print("DATA-EFFICIENCY RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print(
    "\nSaved:",
    output_path
)


# ============================================================
# R² CURVE
# ============================================================

import matplotlib.pyplot as plt

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    results_df["Training_Percentage"],
    results_df["R2"],
    marker="o"
)

plt.xlabel(
    "Training Data Used (%)"
)

plt.ylabel(
    "Test R²"
)

plt.title(
    "Data-Efficiency Curve"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

r2_path = os.path.join(
    FIGURES_DIR,
    "data_efficiency_r2.png"
)

plt.savefig(
    r2_path,
    dpi=300
)

plt.close()


# ============================================================
# RMSE CURVE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    results_df["Training_Percentage"],
    results_df["RMSE"],
    marker="o"
)

plt.xlabel(
    "Training Data Used (%)"
)

plt.ylabel(
    "Test RMSE"
)

plt.title(
    "Data-Efficiency: Prediction Error"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

rmse_path = os.path.join(
    FIGURES_DIR,
    "data_efficiency_rmse.png"
)

plt.savefig(
    rmse_path,
    dpi=300
)

plt.close()


print("\nGenerated:")
print(r2_path)
print(rmse_path)


# ============================================================
# CLEANUP
# ============================================================

del X_full
del y_full
del X_test
del y_test

gc.collect()
