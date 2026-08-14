import os
import gc
import joblib
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


FEATURES_DIR = "features/random"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


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


def prepare_data(df, feature_columns=None):

    y = pd.to_numeric(
        df["SCORE"],
        errors="coerce"
    )

    remove_columns = [
        c
        for c in LEAKAGE_COLUMNS + METADATA_COLUMNS
        if c in df.columns
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

    if feature_columns is not None:
        X = X.reindex(
            columns=feature_columns,
            fill_value=0
        )

    return X, y


print("=" * 70)
print("EXTRA TREES — 50 TREES / DEPTH 25")
print("=" * 70)


# ============================================================
# TRAINING DATA
# ============================================================

print("\nLoading training data...")

train_df = pd.read_csv(
    os.path.join(
        FEATURES_DIR,
        "train_features.csv"
    ),
    low_memory=False
)

X_train, y_train = prepare_data(
    train_df
)

feature_columns = X_train.columns.tolist()

print(
    "Training:",
    X_train.shape
)

del train_df
gc.collect()


# ============================================================
# VALIDATION DATA
# ============================================================

print("\nLoading validation data...")

val_df = pd.read_csv(
    os.path.join(
        FEATURES_DIR,
        "validation_features.csv"
    ),
    low_memory=False
)

X_val, y_val = prepare_data(
    val_df,
    feature_columns
)

print(
    "Validation:",
    X_val.shape
)

del val_df
gc.collect()


# ============================================================
# TEST DATA
# ============================================================

print("\nLoading test data...")

test_df = pd.read_csv(
    os.path.join(
        FEATURES_DIR,
        "test_features.csv"
    ),
    low_memory=False
)

X_test, y_test = prepare_data(
    test_df,
    feature_columns
)

print(
    "Test:",
    X_test.shape
)

del test_df
gc.collect()


# ============================================================
# MODEL
# ============================================================

print("\nTraining Extra Trees...")

model = ExtraTreesRegressor(
    n_estimators=50,
    max_depth=25,
    min_samples_leaf=1,
    max_features=1.0,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# VALIDATION
# ============================================================

print("\nEvaluating validation...")

val_pred = model.predict(
    X_val
)

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

print("\nValidation performance")
print("----------------------")
print(f"RMSE: {val_rmse:.6f}")
print(f"MAE : {val_mae:.6f}")
print(f"R²  : {val_r2:.6f}")


# ============================================================
# TEST
# ============================================================

print("\nEvaluating test...")

test_pred = model.predict(
    X_test
)

test_rmse = mean_squared_error(
    y_test,
    test_pred
) ** 0.5

test_mae = mean_absolute_error(
    y_test,
    test_pred
)

test_r2 = r2_score(
    y_test,
    test_pred
)

print("\nTEST PERFORMANCE")
print("----------------")
print(f"RMSE: {test_rmse:.6f}")
print(f"MAE : {test_mae:.6f}")
print(f"R²  : {test_r2:.6f}")


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "extra_trees_depth25_random.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\nModel saved:")
print(model_path)


# ============================================================
# SAVE RESULTS
# ============================================================

result = pd.DataFrame([{
    "Model": "Extra Trees depth25",
    "n_estimators": 50,
    "max_depth": 25,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "Validation_RMSE": val_rmse,
    "Validation_MAE": val_mae,
    "Validation_R2": val_r2,
    "Test_RMSE": test_rmse,
    "Test_MAE": test_mae,
    "Test_R2": test_r2,
}])

result_path = os.path.join(
    RESULTS_DIR,
    "extra_trees_depth25_random.csv"
)

result.to_csv(
    result_path,
    index=False
)

print("\nResults saved:")
print(result_path)
