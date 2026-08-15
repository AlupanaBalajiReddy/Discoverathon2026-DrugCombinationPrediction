from pathlib import Path
import os
import pandas as pd


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = SUBMISSION_ROOT / "splits"
OUTPUT_DIR = SUBMISSION_ROOT / "features"

DESCRIPTOR_FILE = SUBMISSION_ROOT / "data" / "processed" / "drug_descriptors.csv"


# ============================================================
# LOAD DESCRIPTORS
# ============================================================

print("Loading drug descriptors...")

descriptors = pd.read_csv(
    DESCRIPTOR_FILE,
    low_memory=False
)

descriptors["NSC"] = pd.to_numeric(
    descriptors["NSC"],
    errors="coerce"
).astype("Int64")


print("Descriptor table:", descriptors.shape)


# ============================================================
# DESCRIPTOR COLUMNS
# ============================================================

descriptor_columns = [
    "MolWt",
    "ExactMolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "HeavyAtomCount",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "MolMR"
]


# ============================================================
# FUNCTION TO GENERATE FEATURES
# ============================================================

def generate_features(input_file, output_file):

    print("\nProcessing:")
    print(input_file)

    df = pd.read_csv(
        input_file,
        low_memory=False
    )

    print("Input shape:", df.shape)


    # --------------------------------------------------------
    # Normalize drug IDs
    # --------------------------------------------------------

    df["NSC1"] = pd.to_numeric(
        df["NSC1"],
        errors="coerce"
    ).astype("Int64")

    df["NSC2"] = pd.to_numeric(
        df["NSC2"],
        errors="coerce"
    ).astype("Int64")


    # --------------------------------------------------------
    # Remove target leakage
    # --------------------------------------------------------

    leakage_columns = [
    "PERCENTGROWTH",
    "PERCENTGROWTHNOTZ",
    "TESTVALUE",
    "CONTROLVALUE",
    "TZVALUE",
    "EXPECTEDGROWTH"
]





    existing_leakage_columns = [
        col
        for col in leakage_columns
        if col in df.columns
    ]

    df = df.drop(
        columns=existing_leakage_columns
    )


    # --------------------------------------------------------
    # Drug 1 descriptors
    # --------------------------------------------------------

    desc1 = descriptors[
        ["NSC"] + descriptor_columns
    ].copy()

    desc1 = desc1.rename(
        columns={
            col: f"{col}_1"
            for col in descriptor_columns
        }
    )

    desc1 = desc1.rename(
        columns={
            "NSC": "NSC1"
        }
    )


    df = df.merge(
        desc1,
        on="NSC1",
        how="left"
    )


    # --------------------------------------------------------
    # Drug 2 descriptors
    # --------------------------------------------------------

    desc2 = descriptors[
        ["NSC"] + descriptor_columns
    ].copy()

    desc2 = desc2.rename(
        columns={
            col: f"{col}_2"
            for col in descriptor_columns
        }
    )

    desc2 = desc2.rename(
        columns={
            "NSC": "NSC2"
        }
    )


    df = df.merge(
        desc2,
        on="NSC2",
        how="left"
    )


    # --------------------------------------------------------
    # Check descriptor coverage
    # --------------------------------------------------------

    descriptor_columns_1 = [
        f"{col}_1"
        for col in descriptor_columns
    ]

    descriptor_columns_2 = [
        f"{col}_2"
        for col in descriptor_columns
    ]


    missing_drug1 = df[
        descriptor_columns_1
    ].isna().all(axis=1).sum()


    missing_drug2 = df[
        descriptor_columns_2
    ].isna().all(axis=1).sum()


    print(
        "Rows missing Drug 1 descriptors:",
        missing_drug1
    )

    print(
        "Rows missing Drug 2 descriptors:",
        missing_drug2
    )


    # --------------------------------------------------------
    # Remove rows without descriptors
    # --------------------------------------------------------

    required_descriptor_columns = (
        descriptor_columns_1
        + descriptor_columns_2
    )


    before = len(df)


    df = df.dropna(
        subset=required_descriptor_columns
    )


    after = len(df)


    print(
        "Rows removed because of missing descriptors:",
        before - after
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )


    df.to_csv(
        output_file,
        index=False
    )


    print(
        "Saved:",
        output_file
    )

    print(
        "Final shape:",
        df.shape
    )


# ============================================================
# PROCESS ALL FOUR SPLITS
# ============================================================

split_names = [
    "random",
    "cold_combination",
    "cold_cell_line",
    "cold_drug"
]


for split_name in split_names:

    print("\n")
    print("=" * 70)
    print(f"PROCESSING {split_name.upper()}")
    print("=" * 70)


    input_dir = os.path.join(
        SPLITS_DIR,
        split_name
    )


    output_dir = os.path.join(
        OUTPUT_DIR,
        split_name
    )


    generate_features(
        os.path.join(
            input_dir,
            "train.csv"
        ),
        os.path.join(
            output_dir,
            "train_features.csv"
        )
    )


    generate_features(
        os.path.join(
            input_dir,
            "validation.csv"
        ),
        os.path.join(
            output_dir,
            "validation_features.csv"
        )
    )


    generate_features(
        os.path.join(
            input_dir,
            "test.csv"
        ),
        os.path.join(
            output_dir,
            "test_features.csv"
        )
    )


print("\n")
print("=" * 70)
print("ALL FEATURE DATASETS GENERATED SUCCESSFULLY")
print("=" * 70)
