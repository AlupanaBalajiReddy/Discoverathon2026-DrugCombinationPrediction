import pandas as pd
import os


BASE_DIR = "splits"


def load_split(split_name):
    path = os.path.join(BASE_DIR, split_name)

    train = pd.read_csv(os.path.join(path, "train.csv"), low_memory=False)
    val = pd.read_csv(os.path.join(path, "validation.csv"), low_memory=False)
    test = pd.read_csv(os.path.join(path, "test.csv"), low_memory=False)

    return train, val, test


def get_drugs(df):
    drugs = set(df["NSC1"].astype(str)) | set(df["NSC2"].astype(str))
    drugs.discard("nan")
    return drugs


def check_overlap(a, b):
    return len(a.intersection(b))


# ============================================================
# RANDOM SPLIT
# ============================================================

train, val, test = load_split("random")

print("\n" + "=" * 70)
print("RANDOM SPLIT")
print("=" * 70)

print("Train rows:", len(train))
print("Validation rows:", len(val))
print("Test rows:", len(test))

print("Train-Test pair overlap:",
      len(set(train["drug_pair_id"]) &
          set(test["drug_pair_id"])))


# ============================================================
# COLD-COMBINATION
# ============================================================

train, val, test = load_split("cold_combination")

train_pairs = set(train["drug_pair_id"])
val_pairs = set(val["drug_pair_id"])
test_pairs = set(test["drug_pair_id"])

print("\n" + "=" * 70)
print("COLD-COMBINATION")
print("=" * 70)

print("Train pairs:", len(train_pairs))
print("Validation pairs:", len(val_pairs))
print("Test pairs:", len(test_pairs))

print("Train-Val overlap:",
      len(train_pairs & val_pairs))

print("Train-Test overlap:",
      len(train_pairs & test_pairs))

print("Validation-Test overlap:",
      len(val_pairs & test_pairs))


# ============================================================
# COLD-CELL-LINE
# ============================================================

train, val, test = load_split("cold_cell_line")

train_cells = set(train["CELLNAME"])
val_cells = set(val["CELLNAME"])
test_cells = set(test["CELLNAME"])

print("\n" + "=" * 70)
print("COLD-CELL-LINE")
print("=" * 70)

print("Train cell lines:", len(train_cells))
print("Validation cell lines:", len(val_cells))
print("Test cell lines:", len(test_cells))

print("Train-Val overlap:",
      len(train_cells & val_cells))

print("Train-Test overlap:",
      len(train_cells & test_cells))

print("Validation-Test overlap:",
      len(val_cells & test_cells))


# ============================================================
# COLD-DRUG
# ============================================================

train, val, test = load_split("cold_drug")

assignment = pd.read_csv(
    "splits/cold_drug/drug_assignment.csv"
)

train_drugs = set(
    assignment.loc[
        assignment["split"] == "train",
        "NSC"
    ].astype(str)
)

val_drugs = set(
    assignment.loc[
        assignment["split"] == "validation",
        "NSC"
    ].astype(str)
)

test_drugs = set(
    assignment.loc[
        assignment["split"] == "test",
        "NSC"
    ].astype(str)
)

print("\n" + "=" * 70)
print("COLD-DRUG")
print("=" * 70)

print("Assigned train drugs:", len(train_drugs))
print("Assigned validation drugs:", len(val_drugs))
print("Assigned test drugs:", len(test_drugs))

print("\nAssigned drug overlap:")

print(
    "Train-Val:",
    len(train_drugs & val_drugs)
)

print(
    "Train-Test:",
    len(train_drugs & test_drugs)
)

print(
    "Validation-Test:",
    len(val_drugs & test_drugs)
)

# ------------------------------------------------------------
# Verify that validation rows contain no test drugs
# ------------------------------------------------------------

val_row_drugs = (
    set(val["NSC1"].astype(str)) |
    set(val["NSC2"].astype(str))
)

test_row_drugs = (
    set(test["NSC1"].astype(str)) |
    set(test["NSC2"].astype(str))
)

# Test rows must not contain validation-held-out drugs
val_test_overlap = len(
    val_row_drugs & test_drugs
)

test_val_overlap = len(
    test_row_drugs & val_drugs
)

print("\nValidation rows containing test drugs:",
      val_test_overlap)

print("Test rows containing validation drugs:",
      test_val_overlap)

# ------------------------------------------------------------
# Verify each test row contains at least one test drug
# ------------------------------------------------------------

test_rows_with_heldout_drug = 0

for _, row in test.iterrows():

    d1 = str(row["NSC1"])
    d2 = str(row["NSC2"])

    if d1 in test_drugs or d2 in test_drugs:
        test_rows_with_heldout_drug += 1

print(
    "\nTest rows containing at least one held-out test drug:",
    test_rows_with_heldout_drug,
    "/",
    len(test)
)

# ------------------------------------------------------------
# Final cold-drug check
# ------------------------------------------------------------

cold_drug_ok = (
    len(train_drugs & val_drugs) == 0
    and
    len(train_drugs & test_drugs) == 0
    and
    len(val_drugs & test_drugs) == 0
    and
    val_test_overlap == 0
    and
    test_val_overlap == 0
    and
    test_rows_with_heldout_drug == len(test)
)

print(
    "\nCold-drug split valid:",
    cold_drug_ok
)







# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

print("Cold-combination leakage:",
      len(train_pairs & test_pairs) == 0)

print("Cold-cell-line leakage:",
      len(train_cells & test_cells) == 0)

print("Cold-drug leakage:",
      len(train_drugs & test_drugs) == 0)
