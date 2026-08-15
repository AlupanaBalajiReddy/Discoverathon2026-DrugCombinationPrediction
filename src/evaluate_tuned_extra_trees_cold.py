import os
import gc
import joblib
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


MODEL_DIR = "models"
FEATURES_DIR = "features"
RESULTS_DIR = "results"

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
        c
        for c in remove_columns
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


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


all_results = []


print("=" * 80)
print("TUNED EXTRA TREES — COLD-START EVALUATION")
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

    # ========================================================
    # LOAD TRAINING DATA
    # ========================================================

    print("\nLoading training data...")

    train_df = pd.read_csv(
        train_path,
        low_memory=False
    )

    X_train, y_train = prepare_data(
        train_df
    )

    feature_names = X_train.columns.tolist()

    print(
        "Training:",
        X_train.shape
    )

    del train_df
    gc.collect()

    # ========================================================
    # LOAD VALIDATION DATA
    # ========================================================

    print("\nLoading validation data...")

    val_df = pd.read_csv(
        val_path,
        low_memory=False
    )

    X_val, y_val = prepare_data(
        val_df,
        feature_names
    )

    print(
        "Validation:",
        X_val.shape
    )

    del val_df
    gc.collect()

    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    print("\nLoading test data...")

    test_df = pd.read_csv(
        test_path,
        low_memory=False
    )

    X_test, y_test = prepare_data(
        test_df,
        feature_names
    )

    print(
        "Test:",
        X_test.shape
    )

    del test_df
    gc.collect()

    # ========================================================
    # TRAIN TUNED EXTRA TREES
    # ========================================================

    print("\nTraining tuned Extra Trees...")

    model = ExtraTreesRegressor(
        n_estimators=50,
        max_depth=None,
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

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\nEvaluating validation...")

    val_pred = model.predict(
        X_val
    )

    val_metrics = evaluate(
        y_val,
        val_pred
    )

    print(
        "Validation:",
        val_metrics
    )

    # ========================================================
    # TEST
    # ========================================================

    print("\nEvaluating test...")

    test_pred = model.predict(
        X_test
    )

    test_metrics = evaluate(
        y_test,
        test_pred
    )

    print(
        "Test:",
        test_metrics
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = os.path.join(
        MODEL_DIR,
        f"extra_trees_tuned_{split}.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "\nModel saved:",
        model_path
    )

    # ========================================================
    # STORE RESULTS
    # ========================================================

    all_results.append({
        "Split": split,

        "Validation_RMSE":
            val_metrics["RMSE"],

        "Validation_MAE":
            val_metrics["MAE"],

        "Validation_R2":
            val_metrics["R2"],

        "Test_RMSE":
            test_metrics["RMSE"],

        "Test_MAE":
            test_metrics["MAE"],

        "Test_R2":
            test_metrics["R2"],
    })

    # ========================================================
    # FREE MEMORY
    # ========================================================

    del X_train
    del y_train
    del X_val
    del y_val
    del X_test
    del y_test
    del val_pred
    del test_pred
    del model

    gc.collect()


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

output_path = os.path.join(
    RESULTS_DIR,
    "extra_trees_tuned_cold_splits.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


print("\n" + "=" * 80)
print("FINAL TUNED EXTRA TREES COLD-START RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(output_path)
