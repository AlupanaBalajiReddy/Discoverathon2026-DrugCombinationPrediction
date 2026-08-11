import os
import pandas as pd
import matplotlib.pyplot as plt


os.makedirs("figures", exist_ok=True)


# ============================================================
# LOAD FINAL RESULTS
# ============================================================

df = pd.read_csv(
    "results/final_model_results.csv"
)


# ============================================================
# 1. RANDOM SPLIT MODEL COMPARISON
# ============================================================

random_df = df[
    df["Split"] == "Random"
].sort_values(
    "Test_RMSE"
)


plt.figure(figsize=(10, 6))

plt.bar(
    random_df["Model"],
    random_df["Test_RMSE"]
)

plt.ylabel("Test RMSE")
plt.xlabel("Model")
plt.title(
    "Model Comparison on Random Split"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "figures/final_model_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 2. R² COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    random_df["Model"],
    random_df["Test_R2"]
)

plt.ylabel("Test R²")
plt.xlabel("Model")
plt.title(
    "Model R² Comparison on Random Split"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "figures/final_r2_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 3. COLD-START COMPARISON
# ============================================================

cold_df = df[
    df["Split"].isin([
        "cold_combination",
        "cold_cell_line",
        "cold_drug"
    ])
]


pivot = cold_df.pivot(
    index="Split",
    columns="Model",
    values="Test_RMSE"
)


plt.figure(figsize=(10, 6))

pivot.plot(
    kind="bar",
    ax=plt.gca()
)

plt.ylabel("Test RMSE")
plt.xlabel("Evaluation Setting")
plt.title(
    "Cold-Start Generalization Performance"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "figures/cold_start_rmse.png",
    dpi=300
)

plt.close()


# ============================================================
# 4. COLD-START R²
# ============================================================

pivot_r2 = cold_df.pivot(
    index="Split",
    columns="Model",
    values="Test_R2"
)


plt.figure(figsize=(10, 6))

pivot_r2.plot(
    kind="bar",
    ax=plt.gca()
)

plt.ylabel("Test R²")
plt.xlabel("Evaluation Setting")
plt.title(
    "Cold-Start Generalization — R²"
)

plt.xticks(
    rotation=0
)

plt.axhline(
    0,
    linewidth=1
)

plt.tight_layout()

plt.savefig(
    "figures/cold_start_r2.png",
    dpi=300
)

plt.close()


print("Final figures generated successfully!")

print("\nGenerated:")

print(
    "figures/final_model_comparison.png"
)

print(
    "figures/final_r2_comparison.png"
)

print(
    "figures/cold_start_rmse.png"
)

print(
    "figures/cold_start_r2.png"
)
