from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = SUBMISSION_ROOT / "results"
FIGURES_DIR = SUBMISSION_ROOT / "figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# LOAD COMPREHENSIVE RESULTS
# ============================================================

input_path = os.path.join(
    RESULTS_DIR,
    "comprehensive_metrics.csv"
)

df = pd.read_csv(input_path)

# ============================================================
# RANDOM SPLIT AS REFERENCE
# ============================================================

random_row = df[
    df["Split"] == "random"
].iloc[0]

random_r2 = random_row["R2"]
random_rmse = random_row["RMSE"]
random_mae = random_row["MAE"]
random_pearson = random_row["Pearson"]
random_spearman = random_row["Spearman"]

# ============================================================
# CALCULATE GENERALIZATION GAPS
# ============================================================

df["R2_Gap_vs_Random"] = (
    df["R2"] - random_r2
)

df["RMSE_Change_vs_Random"] = (
    df["RMSE"] - random_rmse
)

df["MAE_Change_vs_Random"] = (
    df["MAE"] - random_mae
)

df["Pearson_Gap_vs_Random"] = (
    df["Pearson"] - random_pearson
)

df["Spearman_Gap_vs_Random"] = (
    df["Spearman"] - random_spearman
)

# ============================================================
# SAVE GENERALIZATION RESULTS
# ============================================================

output_path = os.path.join(
    RESULTS_DIR,
    "generalization_gap.csv"
)

df.to_csv(
    output_path,
    index=False
)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("GENERALIZATION GAP ANALYSIS")
print("=" * 70)

columns = [
    "Split",
    "R2",
    "R2_Gap_vs_Random",
    "RMSE",
    "RMSE_Change_vs_Random",
    "MAE",
    "MAE_Change_vs_Random",
    "Pearson",
    "Pearson_Gap_vs_Random",
    "Spearman",
    "Spearman_Gap_vs_Random"
]

print(
    df[columns].to_string(
        index=False
    )
)

print(
    "\nSaved:",
    output_path
)

# ============================================================
# R² GENERALIZATION FIGURE
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    df["Split"],
    df["R2"]
)

plt.axhline(
    random_r2,
    linestyle="--",
    label="Random split"
)

plt.ylabel("Test R²")
plt.xlabel("Evaluation Split")
plt.title(
    "Generalization Across Evaluation Splits"
)

plt.legend()

plt.tight_layout()

r2_path = os.path.join(
    FIGURES_DIR,
    "generalization_r2.png"
)

plt.savefig(
    r2_path,
    dpi=300
)

plt.close()

# ============================================================
# RMSE GENERALIZATION FIGURE
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    df["Split"],
    df["RMSE"]
)

plt.axhline(
    random_rmse,
    linestyle="--",
    label="Random split"
)

plt.ylabel("Test RMSE")
plt.xlabel("Evaluation Split")
plt.title(
    "Prediction Error Across Evaluation Splits"
)

plt.legend()

plt.tight_layout()

rmse_path = os.path.join(
    FIGURES_DIR,
    "generalization_rmse.png"
)

plt.savefig(
    rmse_path,
    dpi=300
)

plt.close()

print("\nGenerated:")
print(r2_path)
print(rmse_path)
