import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Paths
# ============================================================

RESULTS = Path("submission/results")
FIGURES = Path("submission/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. BASELINE MODEL COMPARISON — MAE
# ============================================================

df = pd.read_csv(
    RESULTS / "model_comparison_random_final.csv"
)

df = df.sort_values("Test_MAE")

plt.figure(figsize=(9, 5.5))

plt.bar(df["Model"], df["Test_MAE"])

plt.ylabel("MAE ↓")
plt.xlabel("Model")
plt.title("Baseline Model Comparison")
plt.xticks(rotation=20)

for i, value in enumerate(df["Test_MAE"]):
    plt.text(
        i,
        value + 0.03,
        f"{value:.3f}",
        ha="center",
        fontsize=10
    )

plt.tight_layout()

plt.savefig(
    FIGURES / "baseline_model_mae.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. BASELINE EXTRA TREES vs FINAL EXTRA TREES
# ============================================================

baseline = pd.read_csv(
    RESULTS / "model_comparison_random_final.csv"
)

baseline_et = baseline[
    baseline["Model"] == "Extra Trees"
].iloc[0]

final = pd.read_csv(
    RESULTS / "comprehensive_metrics.csv"
)

final_et = final[
    final["Split"] == "random"
].iloc[0]

models = [
    "Baseline Extra Trees",
    "Final Extra Trees\n(depth 25)"
]

mae = [
    baseline_et["Test_MAE"],
    final_et["MAE"]
]

plt.figure(figsize=(8, 5.5))

plt.bar(models, mae)

plt.ylabel("MAE ↓")
plt.title("Extra Trees: Baseline vs Final Model")

for i, value in enumerate(mae):
    plt.text(
        i,
        value + 0.03,
        f"{value:.4f}",
        ha="center",
        fontsize=11
    )

improvement = (
    (baseline_et["Test_MAE"] - final_et["MAE"])
    / baseline_et["Test_MAE"]
    * 100
)

plt.figtext(
    0.5,
    0.01,
    f"MAE reduction: {improvement:.2f}%",
    ha="center",
    fontsize=11
)

plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.savefig(
    FIGURES / "extra_trees_optimization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. FINAL EXTRA TREES — MAE ACROSS SPLITS
# ============================================================

df = pd.read_csv(
    RESULTS / "comprehensive_metrics.csv"
)

split_order = [
    "random",
    "cold_cell_line",
    "cold_combination",
    "cold_drug"
]

df["Split"] = pd.Categorical(
    df["Split"],
    categories=split_order,
    ordered=True
)

df = df.sort_values("Split")

labels = [
    "Random",
    "Cold Cell Line",
    "Cold Combination",
    "Cold Drug"
]

plt.figure(figsize=(9, 5.5))

plt.barh(labels, df["MAE"])

plt.xlabel("MAE ↓")
plt.title("Final Extra Trees Performance Across Evaluation Settings")

plt.gca().invert_yaxis()

for i, value in enumerate(df["MAE"]):
    plt.text(
        value + 0.03,
        i,
        f"{value:.4f}",
        va="center",
        fontsize=10
    )

plt.tight_layout()

plt.savefig(
    FIGURES / "final_model_mae_by_split.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. DATA EFFICIENCY — MAE
# ============================================================

df = pd.read_csv(
    RESULTS / "data_efficiency.csv"
)

x = df["Training_Percentage"]
y = df["MAE"]

plt.figure(figsize=(8.5, 5.5))

plt.plot(
    x,
    y,
    marker="o",
    linewidth=2
)

plt.xlabel("Training Data (%)")
plt.ylabel("MAE ↓")
plt.title("Data Efficiency: Training Data vs MAE")

for xi, yi in zip(x, y):
    plt.annotate(
        f"{yi:.3f}",
        (xi, yi),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )

plt.xticks(x)

plt.tight_layout()

plt.savefig(
    FIGURES / "data_efficiency_mae.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. GENERALIZATION GAP — MAE
# ============================================================

df = pd.read_csv(
    RESULTS / "generalization_gap.csv"
)

df = df[df["Split"] != "random"].copy()

labels = [
    "Cold Combination",
    "Cold Cell Line",
    "Cold Drug"
]

# Use the MAE change already calculated in your results file
values = []

for split in [
    "cold_combination",
    "cold_cell_line",
    "cold_drug"
]:
    row = df[df["Split"] == split].iloc[0]
    values.append(row["MAE_Change_vs_Random"])

plt.figure(figsize=(8.5, 5.5))

plt.bar(labels, values)

plt.ylabel("MAE increase vs Random")
plt.title("Cold-Start Generalization Gap")

for i, value in enumerate(values):
    plt.text(
        i,
        value + 0.02,
        f"+{value:.3f}",
        ha="center",
        fontsize=10
    )

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    FIGURES / "generalization_gap_mae.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

print()
print("==========================================")
print("FIGURES GENERATED SUCCESSFULLY")
print("==========================================")

files = [
    "baseline_model_mae.png",
    "extra_trees_optimization.png",
    "final_model_mae_by_split.png",
    "data_efficiency_mae.png",
    "generalization_gap_mae.png"
]

for file in files:
    path = FIGURES / file
    print(f"OK: {path}")

print()
print("All figures saved in:")
print(FIGURES.resolve())
