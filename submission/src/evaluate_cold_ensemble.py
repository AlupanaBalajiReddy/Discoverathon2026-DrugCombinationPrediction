import os
import gc
import joblib
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODEL_DIR = "models"
FEATURES_DIR = "features"

SPLITS = [
    "cold_combination",
    "cold_cell_line",
    "cold_drug",
]

LEAKAGE_COLUMNS = [
    "SCORE",
    "PERCENTGROWTH",
    "PERCENTGROWTHNOTZ",
    "TESTVALUE",
    "CONTROLVALUE",
    "TZVALUE",
    "EXPECTEDGROWTH",
]

METADATA_COLUMNS = [
    "SCREENER",
    "STUDY",
    "TESTDATE",
    "PLATE",
    "PREFIX1",
    "PREFIX2",
    "drug_pair_id",
]

# The random-split validation experiment found these optimal weights.
# They are now FROZEN and will NOT be optimized using cold-start test data.
RF_WEIGHT = 0.5988090406580092
ET_WEIGHT = 0.401190959341986


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
        c for c in remove_columns
        if c in df.columns
    ]

    X = df.drop(columns=remove_columns)

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

    if len(float_columns) > 0:
        X[float_columns] = X[
            float_columns
        ].astype("float32")

    if feature_columns is not None:
        X = X.reindex(
            columns=feature_columns,
            fill_value=0
        )

    return X, y


def get_feature_names(model):

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)

    if hasattr(model, "feature_names_"):
        names = model.feature_names_

        if names is not None:
            return list(names)

    raise RuntimeError(
        f"Could not determine feature names for {type(model).__name__}"
    )


def evaluate(y_true, pred):

    return {
        "RMSE": mean_squared_error(
            y_true,
            pred
        ) ** 0.5,
        "MAE": mean_absolute_error(
            y_true,
            pred
        ),
        "R2": r2_score(
            y_true,
            pred
        )
    }


all_results = []

print("=" * 80)
print("COLD-START RF + EXTRA TREES ENSEMBLE")
print("=" * 80)

for split in SPLITS:

    print("\n" + "=" * 80)
    print("SPLIT:", split)
    print("=" * 80)

    split_features = os.path.join(
        FEATURES_DIR,
        split
    )

    train_path = os.path.join(
        split_features,
        "train_features.csv"
    )

    val_path = os.path.join(
        split_features,
        "validation_features.csv"
    )

    test_path = os.path.join(
        split_features,
        "test_features.csv"
    )

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    rf_path = os.path.join(
        MODEL_DIR,
        f"random_forest_{split}.pkl"
    )

    et_path = os.path.join(
        MODEL_DIR,
        f"extra_trees_{split}.pkl"
    )

    print("\nLoading models...")

    rf = joblib.load(rf_path)
    et = joblib.load(et_path)

    rf_features = get_feature_names(rf)
    et_features = get_feature_names(et)

    print(
        "RF features:",
        len(rf_features)
    )

    print(
        "Extra Trees features:",
        len(et_features)
    )

    # --------------------------------------------------------
    # LOAD VALIDATION
    # --------------------------------------------------------

    print("\nLoading validation data...")

    val_df = pd.read_csv(
        val_path,
        low_memory=False
    )

    X_val_rf, y_val = prepare_data(
        val_df,
        rf_features
    )

    X_val_et, _ = prepare_data(
        val_df,
        et_features
    )

    print(
        "Validation:",
        X_val_rf.shape,
        X_val_et.shape
    )

    rf_val = rf.predict(X_val_rf)
    et_val = et.predict(X_val_et)

    ensemble_val = (
        RF_WEIGHT * rf_val
        + ET_WEIGHT * et_val
    )

    rf_val_metrics = evaluate(
        y_val,
        rf_val
    )

    et_val_metrics = evaluate(
        y_val,
        et_val
    )

    ens_val_metrics = evaluate(
        y_val,
        ensemble_val
    )

    print("\nValidation performance")

    print(
        "Random Forest:",
        rf_val_metrics
    )

    print(
        "Extra Trees:",
        et_val_metrics
    )

    print(
        "Weighted Ensemble:",
        ens_val_metrics
    )

    # --------------------------------------------------------
    # FREE VALIDATION MEMORY
    # --------------------------------------------------------

    del val_df
    del X_val_rf
    del X_val_et
    del y_val
    del rf_val
    del et_val
    del ensemble_val

    gc.collect()

    # --------------------------------------------------------
    # LOAD TEST
    # --------------------------------------------------------

    print("\nLoading test data...")

    test_df = pd.read_csv(
        test_path,
        low_memory=False
    )

    X_test_rf, y_test = prepare_data(
        test_df,
        rf_features
    )

    X_test_et, _ = prepare_data(
        test_df,
        et_features
    )

    print(
        "Test:",
        X_test_rf.shape,
        X_test_et.shape
    )

    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    rf_test = rf.predict(X_test_rf)
    et_test = et.predict(X_test_et)

    ensemble_test = (
        RF_WEIGHT * rf_test
        + ET_WEIGHT * et_test
    )

    rf_test_metrics = evaluate(
        y_test,
        rf_test
    )

    et_test_metrics = evaluate(
        y_test,
        et_test
    )

    ensemble_test_metrics = evaluate(
        y_test,
        ensemble_test
    )

    print("\nTest performance")

    print(
        "Random Forest:",
        rf_test_metrics
    )

    print(
        "Extra Trees:",
        et_test_metrics
    )

    print(
        "Weighted Ensemble:",
        ensemble_test_metrics
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    for model_name, metrics in [
        ("Random Forest", rf_test_metrics),
        ("Extra Trees", et_test_metrics),
        ("RF + Extra Trees Weighted Ensemble",
         ensemble_test_metrics),
    ]:

        all_results.append({
            "Split": split,
            "Model": model_name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "R2": metrics["R2"],
        })

    output_dir = os.path.join(
        "results",
        "ensemble"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    prediction_file = os.path.join(
        output_dir,
        f"{split}_ensemble_predictions.csv"
    )

    pd.DataFrame({
        "SCORE": y_test,
        "random_forest_prediction": rf_test,
        "extra_trees_prediction": et_test,
        "ensemble_prediction": ensemble_test,
    }).to_csv(
        prediction_file,
        index=False
    )

    # --------------------------------------------------------
    # CLEAN MEMORY
    # --------------------------------------------------------

    del test_df
    del X_test_rf
    del X_test_et
    del y_test
    del rf_test
    del et_test
    del ensemble_test
    del rf
    del et

    gc.collect()


# ============================================================
# SAVE SUMMARY
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_path = os.path.join(
    "results",
    "ensemble",
    "cold_ensemble_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\n" + "=" * 80)
print("FINAL COLD-START ENSEMBLE RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)

print("\nFrozen ensemble weights:")
print("Random Forest :", RF_WEIGHT)
print("Extra Trees   :", ET_WEIGHT)

print("\nSaved:")
print(results_path)
