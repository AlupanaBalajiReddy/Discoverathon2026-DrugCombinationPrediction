import os
import gc
import joblib
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


FEATURES_DIR = "features/random"
MODEL_DIR = "models"
RESULTS_DIR = "results"

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
        c for c in
        LEAKAGE_COLUMNS + METADATA_COLUMNS
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


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 70)
print("EXTRA TREES RANDOM-SPLIT TUNING")
print("=" * 70)

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
    "Training matrix:",
    X_train.shape
)

del train_df
gc.collect()


# ============================================================
# LOAD VALIDATION DATA
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
    "Validation matrix:",
    X_val.shape
)

del val_df
gc.collect()


# ============================================================
# CONFIGURATIONS
# ============================================================

configs = [
    {
        "name": "ET_50_depth15",
        "n_estimators": 50,
        "max_depth": 15,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    },
    {
        "name": "ET_50_depth25",
        "n_estimators": 50,
        "max_depth": 25,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    },
    {
        "name": "ET_50_depthNone",
        "n_estimators": 50,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    },
    {
        "name": "ET_100_depth15",
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    },
    {
        "name": "ET_100_depth25",
        "n_estimators": 100,
        "max_depth": 25,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    },
    {
        "name": "ET_100_depth15_leaf2",
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_leaf": 2,
        "max_features": 1.0,
    },
    {
        "name": "ET_100_depth15_sqrt",
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
]


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = []

for config in configs:

    print("\n" + "=" * 70)
    print(
        "Configuration:",
        config["name"]
    )
    print("=" * 70)

    model = ExtraTreesRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        random_state=42,
        n_jobs=-1,
    )

    print("Training...")

    model.fit(
        X_train,
        y_train
    )

    print("Predicting validation set...")

    prediction = model.predict(
        X_val
    )

    rmse = (
        mean_squared_error(
            y_val,
            prediction
        ) ** 0.5
    )

    mae = mean_absolute_error(
        y_val,
        prediction
    )

    r2 = r2_score(
        y_val,
        prediction
    )

    print(
        f"RMSE: {rmse:.6f}"
    )

    print(
        f"MAE : {mae:.6f}"
    )

    print(
        f"R²  : {r2:.6f}"
    )

    results.append({
        "Configuration": config["name"],
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "min_samples_leaf": config["min_samples_leaf"],
        "max_features": str(config["max_features"]),
        "Validation_RMSE": rmse,
        "Validation_MAE": mae,
        "Validation_R2": r2,
    })

    del model
    del prediction
    gc.collect()


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
).sort_values(
    "Validation_R2",
    ascending=False
)

output_path = os.path.join(
    RESULTS_DIR,
    "extra_trees_tuning_random.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 70)
print("TUNING RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(output_path)

print("\nCurrent RF-ET ensemble random test R²:")
print("0.313228")

print(
    "\nDecision threshold: "
    "only configurations with clearly stronger "
    "validation performance will be considered "
    "for further testing."
)
