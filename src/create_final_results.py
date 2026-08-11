import pandas as pd


# ============================================================
# RANDOM SPLIT MODEL COMPARISON
# ============================================================

random_df = pd.read_csv(
    "results/model_comparison_random_final.csv"
)

random_df = random_df[
    [
        "Model",
        "Test_RMSE",
        "Test_MAE",
        "Test_R2"
    ]
]

random_df["Split"] = "Random"


# ============================================================
# RANDOM FOREST COLD SPLITS
# ============================================================

rf_df = pd.read_csv(
    "results/random_forest_all_splits.csv"
)

rf_df = rf_df[
    rf_df["Split"] != "random"
].copy()

rf_df["Model"] = "Random Forest"

rf_df = rf_df[
    [
        "Split",
        "Model",
        "Test_RMSE",
        "Test_MAE",
        "Test_R2"
    ]
]


# ============================================================
# EXTRA TREES COLD SPLITS
# ============================================================

et_df = pd.read_csv(
    "results/extra_trees_cold_splits.csv"
)

et_df["Model"] = "Extra Trees"

et_df = et_df[
    [
        "Split",
        "Model",
        "Test_RMSE",
        "Test_MAE",
        "Test_R2"
    ]
]


# ============================================================
# RANDOM MODEL TABLE
# ============================================================

random_df = random_df[
    [
        "Split",
        "Model",
        "Test_RMSE",
        "Test_MAE",
        "Test_R2"
    ]
]


# ============================================================
# COMBINE
# ============================================================

final_df = pd.concat(
    [
        random_df,
        rf_df,
        et_df
    ],
    ignore_index=True
)


# ============================================================
# SORT
# ============================================================

final_df = final_df.sort_values(
    [
        "Split",
        "Test_RMSE"
    ]
)


# ============================================================
# SAVE
# ============================================================

final_df.to_csv(
    "results/final_model_results.csv",
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 80)
print("FINAL MODEL RESULTS")
print("=" * 80)

print(
    final_df.to_string(
        index=False
    )
)

print()
print(
    "Saved: results/final_model_results.csv"
)
