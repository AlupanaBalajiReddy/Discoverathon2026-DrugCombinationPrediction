import os
import sys
import joblib
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "checkpoints",
    "extra_trees_depth25_random.pkl"
)

LEAKAGE_COLUMNS = [
    "SCORE",
    "PERCENTGROWTH",
    "PERCENTGROWTHNOTZ",
    "TESTVALUE",
    "CONTROLVALUE",
    "TZVALUE",
    "EXPECTEDGROWTH",
]

METADATA_COLUMNS = [
    "SCREENER",
    "STUDY",
    "TESTDATE",
    "PLATE",
    "PREFIX1",
    "PREFIX2",
    "drug_pair_id",
]


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading final Extra Trees model...")

model = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully.")

print(
    f"Number of features: {len(model.feature_names_in_)}"
)


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    remove_columns = [
        c
        for c in LEAKAGE_COLUMNS + METADATA_COLUMNS
        if c in df.columns
    ]

    X = df.drop(
        columns=remove_columns
    )

    categorical_columns = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    if categorical_columns:

        X = pd.get_dummies(
            X,
            columns=categorical_columns,
            drop_first=False,
            dtype="int8"
        )

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.fillna(0)

    # EXACT FEATURE ORDER USED DURING TRAINING
    X = X.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    return X


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(input_file, output_file=None):

    print(
        f"\nLoading input file: {input_file}"
    )

    df = pd.read_csv(
        input_file,
        low_memory=False
    )

    print(
        f"Input shape: {df.shape}"
    )

    X = prepare_data(
        df
    )

    print(
        f"Prediction matrix: {X.shape}"
    )

    predictions = model.predict(
        X
    )

    result = df.copy()

    result["PREDICTED_SCORE"] = predictions

    if output_file is None:

        output_file = os.path.splitext(
            input_file
        )[0] + "_predictions.csv"

    result.to_csv(
        output_file,
        index=False
    )

    print(
        "\nPredictions generated successfully."
    )

    print(
        f"Output: {output_file}"
    )

    print(
        f"Predictions: {len(predictions)}"
    )

    return result


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python submission/src/predict.py <input.csv> [output.csv]"
        )

        print(
            "\nExample:"
        )

        print(
            "python submission/src/predict.py input.csv predictions.csv"
        )

        sys.exit(1)

    input_file = sys.argv[1]

    output_file = (
        sys.argv[2]
        if len(sys.argv) >= 3
        else None
    )

    predict(
        input_file,
        output_file
    )
