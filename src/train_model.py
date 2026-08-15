import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Load dataset
df = pd.read_csv("data/processed/train_features.csv")

print(df.shape)

# Separate features and target
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

print("X shape:", X.shape)
print("y shape:", y.shape)

# -------------------------------
# Drop metadata columns
# -------------------------------

drop_columns = [
    "SCREENER",
    "STUDY",
    "TESTDATE",
    "PLATE",
    "PREFIX1",
    "PREFIX2"
]

X = X.drop(columns=drop_columns)

print("Shape after dropping metadata:", X.shape)

# -------------------------------
# One-hot encode categorical columns
# -------------------------------

categorical_columns = [
    "CONCUNIT1",
    "CONCUNIT2",
    "VALID",
    "PANEL",
    "CELLNAME"
]

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

print("Final feature matrix:", X.shape)

# ---------------------------------
# Train-Test Split
# ---------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain-Test Split Completed")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# ---------------------------------
# Train Random Forest
# ---------------------------------

from sklearn.ensemble import RandomForestRegressor

print("\nTraining Random Forest...")

rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("Random Forest Training Complete!")

# ---------------------------------
# Model Evaluation
# ---------------------------------

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

print("\nMaking predictions...")

y_pred = rf.predict(X_test)

print("Predictions completed!")

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

print("\n==============================")
print("FEATURES USED")
print("==============================")

for col in X.columns:
    print(col)

import joblib

joblib.dump(rf, "models/random_forest.pkl")

print("Model saved successfully!")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    "results/random_forest_feature_importance.csv",
    index=False
)

print("Feature importance saved!")
print(feature_importance.head(20))

import matplotlib.pyplot as plt

top20 = feature_importance.head(20)

plt.figure(figsize=(10, 7))

plt.barh(
    top20["Feature"],
    top20["Importance"]
)

plt.gca().invert_yaxis()

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 20 Important Features")

plt.tight_layout()

plt.savefig(
    "figures/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Feature importance plot saved!")