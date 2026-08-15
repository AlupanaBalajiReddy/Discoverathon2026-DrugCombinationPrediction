import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

df = pd.read_csv(
    "data/processed/train_features.csv",
    low_memory=False
)

# =====================================================
# Development sample for model comparison
# =====================================================

df = df.sample(
    n=300000,
    random_state=42
)

print("Using sample:", df.shape)

print(df.shape)

# Remove target and leakage columns
drop_columns_target = [
    "SCORE",
    "PERCENTGROWTH",
    "PERCENTGROWTHNOTZ",
    "TESTVALUE",
    "CONTROLVALUE",
    "TZVALUE",
    "EXPECTEDGROWTH"
]

X = df.drop(columns=drop_columns_target)
y = df["SCORE"]

drop_columns = [
    "SCREENER",
    "STUDY",
    "TESTDATE",
    "PLATE",
    "PREFIX1",
    "PREFIX2"
]

X = X.drop(columns=drop_columns)

X = pd.get_dummies(
    X,
    columns=[
        "CONCUNIT1",
        "CONCUNIT2",
        "VALID",
        "PANEL",
        "CELLNAME"
    ],
    drop_first=False
)
print("\nRemaining object columns:")
print(X.select_dtypes(include="object").columns.tolist())

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=50,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=200,
        random_state=42
    ),

    "CatBoost": CatBoostRegressor(
        iterations=200,
        random_state=42,
        verbose=False
    )
}

results = []

for name, model in models.items():

    print("=" * 60)
    print(f"Training {name}...")
    print("=" * 60)

    model.fit(X_train, y_train)

    print("Training completed!")

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        model,
        "models/" + filename
    )

    print(f"{filename} saved.")

# ======================================================
# Everything below is OUTSIDE the for loop
# ======================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE"
)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print()
print("=" * 60)
print(results_df)

# ==========================================
# Plot Model Comparison
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.bar(
    results_df["Model"],
    results_df["RMSE"]
)

plt.ylabel("RMSE")
plt.xlabel("Models")
plt.title("Model Comparison")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "figures/model_comparison.png",
    dpi=300
)

plt.show()

print("Model comparison figure saved!")