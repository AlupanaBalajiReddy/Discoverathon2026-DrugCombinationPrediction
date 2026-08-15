import os
import gc
import joblib
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# ============================================================
# SETTINGS
# ============================================================

SPLIT = "random"

FEATURES_DIR = "features"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# COLUMNS
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


# ============================================================
# PREPARE DATA
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
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING RANDOM SPLIT")
print("=" * 70)


train_path = os.path.join(
    FEATURES_DIR,
    SPLIT,
    "train_features.csv"
)

val_path = os.path.join(
    FEATURES_DIR,
    SPLIT,
    "validation_features.csv"
)

test_path = os.path.join(
    FEATURES_DIR,
    SPLIT,
    "test_features.csv"
)


print("\nLoading training data...")

train_df = pd.read_csv(
    train_path,
    low_memory=False
)

print(
    "Training:",
    train_df.shape
)


X_train, y_train = prepare_data(
    train_df
)

del train_df
gc.collect()


print(
    "Training matrix:",
    X_train.shape
)


print("\nLoading validation data...")

val_df = pd.read_csv(
    val_path,
    low_memory=False
)

X_val, y_val = prepare_data(
    val_df,
    feature_columns=X_train.columns
)

del val_df
gc.collect()


print(
    "Validation matrix:",
    X_val.shape
)


print("\nLoading test data...")

test_df = pd.read_csv(
    test_path,
    low_memory=False
)

X_test, y_test = prepare_data(
    test_df,
    feature_columns=X_train.columns
)

del test_df
gc.collect()


print(
    "Test matrix:",
    X_test.shape
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=30,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    ),

    "CatBoost": CatBoostRegressor(
        iterations=100,
        depth=8,
        learning_rate=0.1,
        random_seed=42,
        verbose=False,
        thread_count=-1
    )
}


# ============================================================
# TRAIN
# ============================================================

results = []


for name, model in models.items():

    print("\n")
    print("=" * 70)
    print(f"TRAINING {name.upper()}")
    print("=" * 70)

    model.fit(
        X_train,
        y_train
    )

    print("Training completed!")


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("Validation prediction...")

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


    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print("Test prediction...")

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


    print("\nValidation:")
    print(f"RMSE : {val_rmse:.4f}")
    print(f"MAE  : {val_mae:.4f}")
    print(f"R²   : {val_r2:.4f}")


    print("\nTest:")
    print(f"RMSE : {test_rmse:.4f}")
    print(f"MAE  : {test_mae:.4f}")
    print(f"R²   : {test_r2:.4f}")


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    filename = (
        name.lower()
        .replace(" ", "_")
        + "_random.pkl"
    )

    model_path = os.path.join(
        MODEL_DIR,
        filename
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "Model saved:",
        model_path
    )


    results.append({

        "Model": name,

        "Validation_RMSE": val_rmse,
        "Validation_MAE": val_mae,
        "Validation_R2": val_r2,

        "Test_RMSE": test_rmse,
        "Test_MAE": test_mae,
        "Test_R2": test_r2
    })


    # Free memory

    del model
    del val_pred
    del test_pred

    gc.collect()


# ============================================================
# ADD RANDOM FOREST BASELINE
# ============================================================

results.append({

    "Model": "Random Forest",

    "Validation_RMSE": 10.400914,
    "Validation_MAE": 6.879476,
    "Validation_R2": 0.305330,

    "Test_RMSE": 10.331847,
    "Test_MAE": 6.851900,
    "Test_R2": 0.310151
})


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="Test_RMSE"
)


results_path = os.path.join(
    RESULTS_DIR,
    "model_comparison_random_final.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


print("\n")
print("=" * 70)
print("FINAL MODEL COMPARISON — RANDOM SPLIT")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print(
    "\nResults saved:",
    results_path
)
