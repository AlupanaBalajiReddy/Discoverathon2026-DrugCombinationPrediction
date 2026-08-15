from pathlib import Path
import os
import gc
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# DIRECTORIES
# ============================================================

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]

FEATURES_DIR = SUBMISSION_ROOT / "features"
RESULTS_DIR = SUBMISSION_ROOT / "results"
FIGURES_DIR = SUBMISSION_ROOT / "figures"
CHECKPOINTS_DIR = SUBMISSION_ROOT / "checkpoints"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

FRACTIONS = [0.01, 0.05, 0.10, 0.25]

RANDOM_STATE = 42

N_ESTIMATORS = 50
MAX_DEPTH = 25


# ============================================================
# PREPROCESSING
# Same preprocessing logic used by final model
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
print("MODEL: EXTRA TREES DEPTH-25")
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


print("\nModel configuration:")
print("Algorithm    : Extra Trees Regressor")
print("Trees        :", N_ESTIMATORS)
print("Max depth    :", MAX_DEPTH)
print("Random state :", RANDOM_STATE)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

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
    # EXTRA TREES DEPTH-25
    # --------------------------------------------------------

    model = ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


    print(
        "Training Extra Trees depth-25..."
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
        f"RMSE : {rmse:.6f}"
    )


    print(
        f"MAE  : {mae:.6f}"
    )


    print(
        f"R²   : {r2:.6f}"
    )


    results.append({

        "Training_Fraction":
            fraction,

        "Training_Percentage":
            fraction * 100,

        "Training_Samples":
            n_samples,

        "RMSE":
            rmse,

        "MAE":
            mae,

        "R2":
            r2
    })


    # --------------------------------------------------------
    # MEMORY CLEANUP
    # --------------------------------------------------------

    del X_subset
    del y_subset
    del predictions
    del model

    gc.collect()


# ============================================================
# 100% DATA
#
# Use the actual final Extra Trees depth-25 model
# rather than hard-coded Random Forest metrics.
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATING FINAL EXTRA TREES DEPTH-25 MODEL")
print("100% DATA")
print("=" * 70)


FINAL_MODEL_PATH = os.path.join(
    CHECKPOINTS_DIR,
    "extra_trees_depth25_random.pkl"
)


print(
    "\nLoading final model:",
    FINAL_MODEL_PATH
)


final_model = joblib.load(
    FINAL_MODEL_PATH
)


print(
    "Model:",
    type(final_model).__name__
)


print(
    "Trees:",
    final_model.n_estimators
)


print(
    "Max depth:",
    final_model.max_depth
)


print(
    "Features:",
    len(final_model.feature_names_in_)
)


# ------------------------------------------------------------
# ALIGN TEST FEATURES EXACTLY TO FINAL MODEL
# ------------------------------------------------------------

X_test_final = X_test.reindex(
    columns=final_model.feature_names_in_,
    fill_value=0
)


print(
    "Final prediction matrix:",
    X_test_final.shape
)


# ------------------------------------------------------------
# FINAL PREDICTION
# ------------------------------------------------------------

final_predictions = final_model.predict(
    X_test_final
)


# ------------------------------------------------------------
# FINAL METRICS
# ------------------------------------------------------------

final_rmse = mean_squared_error(
    y_test,
    final_predictions
) ** 0.5


final_mae = mean_absolute_error(
    y_test,
    final_predictions
)


final_r2 = r2_score(
    y_test,
    final_predictions
)


print(
    f"Final RMSE : {final_rmse:.6f}"
)


print(
    f"Final MAE  : {final_mae:.6f}"
)


print(
    f"Final R²   : {final_r2:.6f}"
)


results.append({

    "Training_Fraction":
        1.0,

    "Training_Percentage":
        100.0,

    "Training_Samples":
        len(X_full),

    "RMSE":
        final_rmse,

    "MAE":
        final_mae,

    "R2":
        final_r2
})


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


# ============================================================
# PRINT RESULTS
# ============================================================

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


print("\nExperiment completed successfully.")
