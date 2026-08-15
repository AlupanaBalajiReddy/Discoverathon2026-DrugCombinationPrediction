import os
import gc
import joblib
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ============================================================
# CONFIGURATION
# ============================================================

FEATURES_DIR = "features/random"
MODEL_DIR = "models"
OUTPUT_DIR = "results/ensemble"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    "Random Forest": "random_forest_random.pkl",
    "Extra Trees": "extra_trees_random.pkl",
    "XGBoost": "xgboost_random.pkl",
    "LightGBM": "lightgbm_random.pkl",
    "CatBoost": "catboost_random.pkl",
}

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


# ============================================================
# FEATURE PREPARATION
# ============================================================

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


# ============================================================
# GET MODEL FEATURE NAMES
# ============================================================

def get_feature_names(model):

    # Standard sklearn models
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # LightGBM
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)

    # CatBoost
    if hasattr(model, "feature_names_"):
        names = model.feature_names_

        if names is not None:
            return list(names)

    raise RuntimeError(
        f"Could not determine feature names for "
        f"{type(model).__name__}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ENSEMBLE EVALUATION")
print("=" * 70)

print("\nLoading validation data...")

val_df = pd.read_csv(
    os.path.join(
        FEATURES_DIR,
        "validation_features.csv"
    ),
    low_memory=False
)

print("Validation shape:", val_df.shape)

print("\nLoading test data...")

test_df = pd.read_csv(
    os.path.join(
        FEATURES_DIR,
        "test_features.csv"
    ),
    low_memory=False
)

print("Test shape:", test_df.shape)

y_val = pd.to_numeric(
    val_df["SCORE"],
    errors="coerce"
).to_numpy()

y_test = pd.to_numeric(
    test_df["SCORE"],
    errors="coerce"
).to_numpy()


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

val_predictions = {}
test_predictions = {}

for model_name, filename in MODELS.items():

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    print("Loading:", path)

    model = joblib.load(path)

    print(
        "Model type:",
        type(model).__name__
    )

    feature_names = get_feature_names(model)

    print(
        "Number of features:",
        len(feature_names)
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    X_val, _ = prepare_data(
        val_df,
        feature_columns=feature_names
    )

    print(
        "Validation matrix:",
        X_val.shape
    )

    val_pred = model.predict(
        X_val
    )

    val_predictions[model_name] = np.asarray(
        val_pred,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    X_test, _ = prepare_data(
        test_df,
        feature_columns=feature_names
    )

    print(
        "Test matrix:",
        X_test.shape
    )

    test_pred = model.predict(
        X_test
    )

    test_predictions[model_name] = np.asarray(
        test_pred,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # INDIVIDUAL VALIDATION PERFORMANCE
    # --------------------------------------------------------

    val_rmse = mean_squared_error(
        y_val,
        val_pred
    ) ** 0.5

    val_mae = mean_absolute_error(
        y_val,
        val_pred
    )

    val_r2 = r2_score(
        y_val,
        val_pred
    )

    print(
        f"Validation RMSE: {val_rmse:.6f}"
    )

    print(
        f"Validation MAE : {val_mae:.6f}"
    )

    print(
        f"Validation R²  : {val_r2:.6f}"
    )

    del X_val
    del X_test
    del model

    gc.collect()


# ============================================================
# CONVERT PREDICTIONS TO MATRICES
# ============================================================

model_names = list(MODELS.keys())

P_val = np.column_stack(
    [
        val_predictions[name]
        for name in model_names
    ]
)

P_test = np.column_stack(
    [
        test_predictions[name]
        for name in model_names
    ]
)

print("\nPrediction matrices:")
print("Validation:", P_val.shape)
print("Test      :", P_test.shape)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_prediction(
    name,
    y_true,
    prediction
):

    rmse = mean_squared_error(
        y_true,
        prediction
    ) ** 0.5

    mae = mean_absolute_error(
        y_true,
        prediction
    )

    r2 = r2_score(
        y_true,
        prediction
    )

    return {
        "Ensemble": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


# ============================================================
# EQUAL-WEIGHT ENSEMBLE
# ============================================================

equal_weights = np.ones(
    len(model_names)
) / len(model_names)

equal_val_pred = P_val @ equal_weights
equal_test_pred = P_test @ equal_weights

results = []

results.append(
    {
        "Ensemble": "Equal Weight",
        "RMSE": mean_squared_error(
            y_test,
            equal_test_pred
        ) ** 0.5,
        "MAE": mean_absolute_error(
            y_test,
            equal_test_pred
        ),
        "R2": r2_score(
            y_test,
            equal_test_pred
        ),
    }
)

# ============================================================
# PAIRWISE ENSEMBLES WITH RANDOM FOREST
# ============================================================

rf_index = model_names.index(
    "Random Forest"
)

for model_name in model_names:

    if model_name == "Random Forest":
        continue

    other_index = model_names.index(
        model_name
    )

    pair_val = (
        P_val[:, rf_index]
        + P_val[:, other_index]
    ) / 2

    pair_test = (
        P_test[:, rf_index]
        + P_test[:, other_index]
    ) / 2

    results.append(
        {
            "Ensemble":
                f"RF + {model_name}",
            "RMSE":
                mean_squared_error(
                    y_test,
                    pair_test
                ) ** 0.5,
            "MAE":
                mean_absolute_error(
                    y_test,
                    pair_test
                ),
            "R2":
                r2_score(
                    y_test,
                    pair_test
                ),
        }
    )


# ============================================================
# VALIDATION-OPTIMIZED WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("OPTIMIZING ENSEMBLE WEIGHTS USING VALIDATION SET")
print("=" * 70)


def objective(weights):

    prediction = P_val @ weights

    return mean_squared_error(
        y_val,
        prediction
    )


constraints = [
    {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }
]

bounds = [
    (0.0, 1.0)
    for _ in model_names
]

initial_weights = np.ones(
    len(model_names)
) / len(model_names)

optimization = minimize(
    objective,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={
        "maxiter": 500,
        "ftol": 1e-12
    }
)

if not optimization.success:

    raise RuntimeError(
        "Weight optimization failed: "
        + optimization.message
    )

optimized_weights = optimization.x

print("\nOptimized weights:")

for name, weight in zip(
    model_names,
    optimized_weights
):

    print(
        f"{name:15s}: {weight:.6f}"
    )

print(
    "Weight sum:",
    optimized_weights.sum()
)


# ============================================================
# OPTIMIZED ENSEMBLE
# ============================================================

weighted_val_pred = (
    P_val @ optimized_weights
)

weighted_test_pred = (
    P_test @ optimized_weights
)

weighted_val_rmse = (
    mean_squared_error(
        y_val,
        weighted_val_pred
    ) ** 0.5
)

weighted_val_mae = (
    mean_absolute_error(
        y_val,
        weighted_val_pred
    )
)

weighted_val_r2 = r2_score(
    y_val,
    weighted_val_pred
)

weighted_test_rmse = (
    mean_squared_error(
        y_test,
        weighted_test_pred
    ) ** 0.5
)

weighted_test_mae = (
    mean_absolute_error(
        y_test,
        weighted_test_pred
    )
)

weighted_test_r2 = r2_score(
    y_test,
    weighted_test_pred
)

print("\nValidation performance:")
print(
    f"RMSE: {weighted_val_rmse:.6f}"
)
print(
    f"MAE : {weighted_val_mae:.6f}"
)
print(
    f"R²  : {weighted_val_r2:.6f}"
)

print("\nTest performance:")
print(
    f"RMSE: {weighted_test_rmse:.6f}"
)
print(
    f"MAE : {weighted_test_mae:.6f}"
)
print(
    f"R²  : {weighted_test_r2:.6f}"
)


# ============================================================
# SAVE WEIGHTS
# ============================================================

weights_df = pd.DataFrame(
    {
        "Model": model_names,
        "Weight": optimized_weights
    }
)

weights_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ensemble_weights.csv"
    ),
    index=False
)


# ============================================================
# SAVE TEST RESULTS
# ============================================================

results.append(
    {
        "Ensemble":
            "Validation-Weighted Ensemble",
        "RMSE":
            weighted_test_rmse,
        "MAE":
            weighted_test_mae,
        "R2":
            weighted_test_r2,
    }
)

results_df = pd.DataFrame(
    results
).sort_values(
    "R2",
    ascending=False
)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ensemble_comparison.csv"
    ),
    index=False
)


# ============================================================
# SAVE VALIDATION RESULTS
# ============================================================

validation_results = []

for i, name in enumerate(model_names):

    prediction = P_val[:, i]

    validation_results.append(
        {
            "Model": name,
            "RMSE":
                mean_squared_error(
                    y_val,
                    prediction
                ) ** 0.5,
            "MAE":
                mean_absolute_error(
                    y_val,
                    prediction
                ),
            "R2":
                r2_score(
                    y_val,
                    prediction
                ),
        }
    )

validation_results.append(
    {
        "Model":
            "Equal Weight Ensemble",
        "RMSE":
            mean_squared_error(
                y_val,
                equal_val_pred
            ) ** 0.5,
        "MAE":
            mean_absolute_error(
                y_val,
                equal_val_pred
            ),
        "R2":
            r2_score(
                y_val,
                equal_val_pred
            ),
    }
)

validation_results.append(
    {
        "Model":
            "Validation-Weighted Ensemble",
        "RMSE":
            weighted_val_rmse,
        "MAE":
            weighted_val_mae,
        "R2":
            weighted_val_r2,
    }
)

pd.DataFrame(
    validation_results
).sort_values(
    "R2",
    ascending=False
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ensemble_validation.csv"
    ),
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

pd.DataFrame(
    {
        "y_true": y_test,
        "ensemble_prediction": weighted_test_pred
    }
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "weighted_ensemble_test_predictions.csv"
    ),
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL ENSEMBLE COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nFiles saved to:")
print(OUTPUT_DIR)

print("\nEnsemble evaluation complete.")
