import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_FILE = "data/raw/ComboDrugGrowth_Nov2017.csv"
OUTPUT_DIR = "splits"

RANDOM_STATE = 42

TRAIN_SIZE = 0.80
VAL_SIZE = 0.10
TEST_SIZE = 0.10


def make_canonical_pair(row):
    drugs = sorted([str(row["NSC1"]), str(row["NSC2"])])
    return f"{drugs[0]}_{drugs[1]}"


def save_split(df, name):
    output_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(output_dir, exist_ok=True)

    train, temp = train_test_split(
        df,
        test_size=VAL_SIZE + TEST_SIZE,
        random_state=RANDOM_STATE
    )

    val, test = train_test_split(
        temp,
        test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE
    )

    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"\n{name}")
    print("-" * 50)
    print("Train:", train.shape)
    print("Validation:", val.shape)
    print("Test:", test.shape)


def group_split(df, group_column, name):
    groups = df[group_column].drop_duplicates()

    train_groups, temp_groups = train_test_split(
        groups,
        test_size=VAL_SIZE + TEST_SIZE,
        random_state=RANDOM_STATE
    )

    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE
    )

    train = df[df[group_column].isin(train_groups)].copy()
    val = df[df[group_column].isin(val_groups)].copy()
    test = df[df[group_column].isin(test_groups)].copy()

    output_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(output_dir, exist_ok=True)

    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"\n{name}")
    print("-" * 50)
    print("Train:", train.shape)
    print("Validation:", val.shape)
    print("Test:", test.shape)

    print("Unique groups:")
    print("Train:", train[group_column].nunique())
    print("Validation:", val[group_column].nunique())
    print("Test:", test[group_column].nunique())


# ============================================================
# LOAD DATA
# ============================================================

print("Loading NCI-ALMANAC dataset...")

df = pd.read_csv(
    RAW_FILE,
    low_memory=False
)

print("Raw shape:", df.shape)


# ============================================================
# REMOVE ROWS WITHOUT TARGET
# ============================================================

df = df.dropna(subset=["SCORE"]).copy()

print("After removing missing SCORE:", df.shape)

# Normalize drug identifiers
df["NSC1"] = pd.to_numeric(df["NSC1"], errors="coerce").astype("Int64")
df["NSC2"] = pd.to_numeric(df["NSC2"], errors="coerce").astype("Int64")

# Remove rows where drug IDs could not be converted
df = df.dropna(subset=["NSC1", "NSC2"]).copy()

# Convert to consistent string representation
df["NSC1"] = df["NSC1"].astype(str)
df["NSC2"] = df["NSC2"].astype(str)

print("After normalizing drug IDs:", df.shape)

# ============================================================
# CANONICAL DRUG PAIR
# ============================================================

df["drug_pair_id"] = df.apply(
    make_canonical_pair,
    axis=1
)

print("Unique drug pairs:", df["drug_pair_id"].nunique())
print("Unique cell lines:", df["CELLNAME"].nunique())


# ============================================================
# 1. RANDOM SPLIT
# ============================================================

save_split(
    df,
    "random"
)


# ============================================================
# 2. COLD-COMBINATION SPLIT
# ============================================================

group_split(
    df,
    "drug_pair_id",
    "cold_combination"
)


# ============================================================
# 3. COLD-CELL-LINE SPLIT
# ============================================================

group_split(
    df,
    "CELLNAME",
    "cold_cell_line"
)


# ============================================================
# 4. COLD-DRUG SPLIT
# ============================================================

print("\nGenerating cold-drug split...")

# Get all unique normalized drugs
all_drugs = sorted(
    set(df["NSC1"].astype(str)) |
    set(df["NSC2"].astype(str))
)

rng = np.random.RandomState(RANDOM_STATE)
rng.shuffle(all_drugs)

n = len(all_drugs)

n_train = int(n * TRAIN_SIZE)
n_val = int(n * VAL_SIZE)

train_drugs = set(all_drugs[:n_train])
val_drugs = set(all_drugs[n_train:n_train + n_val])
test_drugs = set(all_drugs[n_train + n_val:])

print("\nDrug groups:")
print("Train drugs:", len(train_drugs))
print("Validation drugs:", len(val_drugs))
print("Test drugs:", len(test_drugs))

# Save the official cold-drug group assignment
drug_assignment = pd.DataFrame({
    "NSC": (
        list(train_drugs)
        + list(val_drugs)
        + list(test_drugs)
    ),
    "split": (
        ["train"] * len(train_drugs)
        + ["validation"] * len(val_drugs)
        + ["test"] * len(test_drugs)
    )
})

os.makedirs(
    os.path.join(OUTPUT_DIR, "cold_drug"),
    exist_ok=True
)

drug_assignment.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cold_drug",
        "drug_assignment.csv"
    ),
    index=False
)

print(
    "Drug assignment saved to "
    "splits/cold_drug/drug_assignment.csv"
)




def assign_cold_drug_split(row):

    d1 = str(row["NSC1"])
    d2 = str(row["NSC2"])

    # --------------------------------------------------------
    # TRAIN
    # Both drugs must be training drugs
    # --------------------------------------------------------

    if d1 in train_drugs and d2 in train_drugs:
        return "train"

    # --------------------------------------------------------
    # VALIDATION
    # At least one validation drug
    # AND no test drug
    # --------------------------------------------------------

    if (
        (d1 in val_drugs or d2 in val_drugs)
        and
        (d1 not in test_drugs and d2 not in test_drugs)
    ):
        return "validation"

    # --------------------------------------------------------
    # TEST
    # At least one test drug
    # AND no validation drug
    # --------------------------------------------------------

    if (
        (d1 in test_drugs or d2 in test_drugs)
        and
        (d1 not in val_drugs and d2 not in val_drugs)
    ):
        return "test"

    # --------------------------------------------------------
    # Cross-validation/test combinations are discarded
    # --------------------------------------------------------

    return "discard"


df["drug_split"] = df.apply(
    assign_cold_drug_split,
    axis=1
)


train = df[
    df["drug_split"] == "train"
].drop(
    columns=["drug_split"]
)

val = df[
    df["drug_split"] == "validation"
].drop(
    columns=["drug_split"]
)

test = df[
    df["drug_split"] == "test"
].drop(
    columns=["drug_split"]
)


output_dir = os.path.join(
    OUTPUT_DIR,
    "cold_drug"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


train.to_csv(
    os.path.join(output_dir, "train.csv"),
    index=False
)

val.to_csv(
    os.path.join(output_dir, "validation.csv"),
    index=False
)

test.to_csv(
    os.path.join(output_dir, "test.csv"),
    index=False
)


print("\ncold_drug")
print("-" * 50)

print("Train:", train.shape)
print("Validation:", val.shape)
print("Test:", test.shape)

print("\nDrugs appearing in each split:")

print(
    "Train:",
    len(
        set(train["NSC1"].astype(str)) |
        set(train["NSC2"].astype(str))
    )
)

print(
    "Validation:",
    len(
        set(val["NSC1"].astype(str)) |
        set(val["NSC2"].astype(str))
    )
)

print(
    "Test:",
    len(
        set(test["NSC1"].astype(str)) |
        set(test["NSC2"].astype(str))
    )
)

print("\nCold-drug split complete.")

print("\nAll splits generated successfully!")
