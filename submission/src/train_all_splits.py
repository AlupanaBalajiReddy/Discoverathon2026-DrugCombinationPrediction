import os
import gc
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


FEATURES_DIR = "features"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


SPLITS = [
    "random",
    "cold_combination",
    "cold_cell_line",
    "cold_drug"
]


# ============================================================
# COLUMNS TO REMOVE
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

    # Convert to numeric
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Fill missing values
    X = X.fillna(0)

    # Convert float64 → float32
    float_columns = X.select_dtypes(
        include=["float64"]
    ).columns

    X[float_columns] = X[
        float_columns
    ].astype("float32")

    # Align validation/test to training features
    if feature_columns is not None:

        X = X.reindex(
            columns=feature_columns,
            fill_value=0
        )

    return X, y


# ============================================================
# TRAIN ONE SPLIT
# ============================================================

def train_split(split_name):

    print("\n")
    print("=" * 70)
    print(f"TRAINING SPLIT: {split_name.upper()}")
    print("=" * 70)


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_path = os.path.join(
        FEATURES_DIR,
        split_name,
        "train_features.csv"
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

    X_train, y_train = prepare_data(
        train_df
    )

    del train_df
    gc.collect()

    print(
        "Training matrix:",
        X_train.shape
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_path = os.path.join(
        FEATURES_DIR,
        split_name,
        "validation_features.csv"
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


    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    model = RandomForestRegressor(
        n_estimators=30,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print("Random Forest training complete!")


    # --------------------------------------------------------
    # VALIDATION PREDICTION
    # --------------------------------------------------------

    print("\nMaking validation predictions...")

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

    print("\nValidation:")
    print(f"RMSE : {val_rmse:.4f}")
    print(f"MAE  : {val_mae:.4f}")
    print(f"R²   : {val_r2:.4f}")


    # Free validation memory
    del X_val
    del y_val
    del val_pred
    gc.collect()


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_path = os.path.join(
        FEATURES_DIR,
        split_name,
        "test_features.csv"
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


    # --------------------------------------------------------
    # TEST PREDICTION
    # --------------------------------------------------------

    print("\nMaking test predictions...")

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

    print("\nTest:")
    print(f"RMSE : {test_rmse:.4f}")
    print(f"MAE  : {test_mae:.4f}")
    print(f"R²   : {test_r2:.4f}")


    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        f"random_forest_{split_name}.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "\nModel saved:",
        model_path
    )


    # --------------------------------------------------------
    # CLEAN MEMORY
    # --------------------------------------------------------

    del X_train
    del y_train
    del X_test
    del y_test
    del test_pred
    del model

    gc.collect()


    return {
        "Split": split_name,
        "Validation_RMSE": val_rmse,
        "Validation_MAE": val_mae,
        "Validation_R2": val_r2,
        "Test_RMSE": test_rmse,
        "Test_MAE": test_mae,
        "Test_R2": test_r2
    }


# ============================================================
# MAIN
# ============================================================

results = []


for split_name in SPLITS:

    result = train_split(
        split_name
    )

    results.append(
        result
    )

    gc.collect()


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_path = os.path.join(
    RESULTS_DIR,
    "random_forest_all_splits.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


print("\n")
print("=" * 70)
print("FINAL RANDOM FOREST RESULTS")
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
