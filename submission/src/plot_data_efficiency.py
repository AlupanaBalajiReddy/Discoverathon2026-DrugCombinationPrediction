from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# DIRECTORIES
# ============================================================

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = SUBMISSION_ROOT / "results"
FIGURES_DIR = SUBMISSION_ROOT / "figures"

os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# LOAD RESULTS
# ============================================================

input_path = os.path.join(
    RESULTS_DIR,
    "data_efficiency.csv"
)

df = pd.read_csv(input_path)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 70)
print("DATA-EFFICIENCY PLOTTING")
print("=" * 70)

print(
    df.to_string(index=False)
)


# ============================================================
# R² FIGURE
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    df["Training_Percentage"],
    df["R2"],
    marker="o",
    linewidth=2
)

plt.xlabel(
    "Training Data Used (%)"
)

plt.ylabel(
    "Test R²"
)

plt.title(
    "Data Efficiency of Extra Trees Depth-25 Model"
)

plt.xticks(
    df["Training_Percentage"]
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


r2_path = os.path.join(
    FIGURES_DIR,
    "data_efficiency_r2.png"
)

plt.savefig(
    r2_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# RMSE FIGURE
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    df["Training_Percentage"],
    df["RMSE"],
    marker="o",
    linewidth=2
)

plt.xlabel(
    "Training Data Used (%)"
)

plt.ylabel(
    "Test RMSE"
)

plt.title(
    "Prediction Error vs Training Data"
)

plt.xticks(
    df["Training_Percentage"]
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


rmse_path = os.path.join(
    FIGURES_DIR,
    "data_efficiency_rmse.png"
)

plt.savefig(
    rmse_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# DONE
# ============================================================

print()
print("Generated:")
print(r2_path)
print(rmse_path)
