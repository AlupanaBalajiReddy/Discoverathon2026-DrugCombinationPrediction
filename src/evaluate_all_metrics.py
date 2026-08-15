import os
import gc
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)

from scipy.stats import pearsonr, spearmanr


# ============================================================
# DIRECTORIES
# ============================================================

FEATURES_DIR = "features"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# SPLITS
# ============================================================

SPLITS = [
    "random",
    "cold_combination",
    "cold_cell_line",
    "cold_drug"
]


# ============================================================
# DATA PREPROCESSING
# SAME LOGIC USED DURING TRAINING
# ============================================================

LEAKAGE_COLUMNS = [
    "SCORE",
    "PERCENTGROWTH",
    "PERCENTGROWTHNOTZ",
    "TESTVALUE",
    "CONTROLVALUE",
    "TZVALUE",
    "EXPECTEDGROWTH"
]


METADATA_COLUMNS = [
    "SCREENER",
    "STUDY",
    "TESTDATE",
    "PLATE",
    "PREFIX1",
    "PREFIX2",
    "drug_pair_id"
]


def prepare_data(df, feature_columns=None):

    y = pd.to_numeric(
        df["SCORE"],
        errors="coerce"
    )

    remove_columns = (
        LEAKAGE_COLUMNS
        + METADATA_COLUMNS
    )

    remove_columns = [
        col
        for col in remove_columns
        if col in df.columns
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

    float_columns = X.select_dtypes(
        include=["float64"]
    ).columns

    X[float_columns] = X[
        float_columns
    ].astype("float32")

    if feature_columns is not None:

        X = X.reindex(
            columns=feature_columns,
            fill_value=0
        )

    return X, y


# ============================================================
# CORRELATION
# ============================================================

def calculate_correlations(y_true, y_pred):

    pearson = pearsonr(
        y_true,
        y_pred
    ).statistic

    spearman = spearmanr(
        y_true,
        y_pred
    ).statistic

    return pearson, spearman


# ============================================================
# CLASSIFICATION METRICS
#
# Synergy hit = top 10% of TRAINING SCORE distribution
#
# This prevents the test set from determining its own threshold.
# ============================================================

def calculate_classification_metrics(
    y_train,
    y_test,
    y_pred
):

    threshold = np.quantile(
        y_train,
        0.90
    )

    y_true_binary = (
        y_test >= threshold
    ).astype(int)

    y_pred_binary = (
        y_pred >= threshold
    ).astype(int)

    macro_f1 = f1_score(
        y_true_binary,
        y_pred_binary,
        average="macro",
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        y_true_binary,
        y_pred_binary
    )

    try:
        auroc = roc_auc_score(
            y_true_binary,
            y_pred
        )
    except ValueError:
        auroc = np.nan

    try:
        auprc = average_precision_score(
            y_true_binary,
            y_pred
        )
    except ValueError:
        auprc = np.nan

    try:
        mcc = matthews_corrcoef(
            y_true_binary,
            y_pred_binary
        )
    except ValueError:
        mcc = np.nan

    return (
        threshold,
        macro_f1,
        balanced_acc,
        auroc,
        auprc,
        mcc
    )


# ============================================================
# RANKING METRICS
#
# Ranking is performed within each CELLNAME.
#
# For each cell line:
#   - rank predictions
#   - identify actual top 10% as relevant
#   - calculate Precision@50
#   - calculate Recall@100
#   - calculate nDCG@100
#   - calculate Enrichment@100
#
# Final value = mean across cell lines.
# ============================================================

def dcg_at_k(relevances, k):

    relevances = np.asarray(
        relevances[:k]
    )

    if len(relevances) == 0:
        return 0.0

    discounts = np.log2(
        np.arange(
            2,
            len(relevances) + 2
        )
    )

    return np.sum(
        (2 ** relevances - 1)
        / discounts
    )


def ndcg_at_k(relevances, k):

    actual_dcg = dcg_at_k(
        relevances,
        k
    )

    ideal_relevances = sorted(
        relevances,
        reverse=True
    )

    ideal_dcg = dcg_at_k(
        ideal_relevances,
        k
    )

    if ideal_dcg == 0:
        return np.nan

    return actual_dcg / ideal_dcg


def calculate_ranking_metrics(
    test_df,
    y_test,
    y_pred
):

    ranking_df = test_df.copy()

    ranking_df["TRUE_SCORE"] = np.asarray(
        y_test
    )

    ranking_df["PRED_SCORE"] = np.asarray(
        y_pred
    )

    precision_values = []
    recall_values = []
    ndcg_values = []
    enrichment_values = []

    # --------------------------------------------------------
    # Rank independently within each cancer cell line
    # --------------------------------------------------------

    for cell_line, group in ranking_df.groupby(
        "CELLNAME"
    ):

        if len(group) < 100:
            continue

        group = group.copy()

        # Actual top 10% = relevant
        threshold = group[
            "TRUE_SCORE"
        ].quantile(0.90)

        group["RELEVANT"] = (
            group["TRUE_SCORE"]
            >= threshold
        ).astype(int)

        # Rank by prediction
        ranked = group.sort_values(
            "PRED_SCORE",
            ascending=False
        )

        # ----------------------------------------------------
        # Precision@50
        # ----------------------------------------------------

        top50 = ranked.head(50)

        precision50 = (
            top50["RELEVANT"].sum()
            / 50
        )

        precision_values.append(
            precision50
        )

        # ----------------------------------------------------
        # Recall@100
        # ----------------------------------------------------

        top100 = ranked.head(100)

        total_relevant = (
            group["RELEVANT"].sum()
        )

        if total_relevant > 0:

            recall100 = (
                top100["RELEVANT"].sum()
                / total_relevant
            )

            recall_values.append(
                recall100
            )

        # ----------------------------------------------------
        # nDCG@100
        # ----------------------------------------------------

        ndcg = ndcg_at_k(
            top100["RELEVANT"].tolist(),
            100
        )

        if not np.isnan(ndcg):

            ndcg_values.append(
                ndcg
            )

        # ----------------------------------------------------
        # Enrichment@100
        # ----------------------------------------------------

        observed_hits = (
            top100["RELEVANT"].sum()
        )

        expected_hits = (
            100
            * total_relevant
            / len(group)
        )

        if expected_hits > 0:

            enrichment = (
                observed_hits
                / expected_hits
            )

            enrichment_values.append(
                enrichment
            )

    return (
        np.mean(precision_values)
        if precision_values else np.nan,

        np.mean(recall_values)
        if recall_values else np.nan,

        np.mean(ndcg_values)
        if ndcg_values else np.nan,

        np.mean(enrichment_values)
        if enrichment_values else np.nan
    )


# ============================================================
# EVALUATE ONE SPLIT
# ============================================================

def evaluate_split(split_name):

    print("\n")
    print("=" * 70)
    print(
        f"EVALUATING: {split_name.upper()}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # LOAD TRAINING DATA
    # Used for:
    #   1. feature columns
    #   2. synergy threshold
    # --------------------------------------------------------

    train_path = os.path.join(
        FEATURES_DIR,
        split_name,
        "train_features.csv"
    )

    print("\nLoading training data...")

    train_df = pd.read_csv(
        train_path,
        low_memory=False
    )

    X_train, y_train = prepare_data(
        train_df
    )

    y_train = y_train.dropna()

    feature_columns = X_train.columns

    print(
        "Training matrix:",
        X_train.shape
    )

    # We don't need X_train anymore
    del X_train
    gc.collect()


    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    test_path = os.path.join(
        FEATURES_DIR,
        split_name,
        "test_features.csv"
    )

    print("\nLoading test data...")

    test_df = pd.read_csv(
        test_path,
        low_memory=False
    )

    X_test, y_test = prepare_data(
        test_df,
        feature_columns=feature_columns
    )

    print(
        "Test matrix:",
        X_test.shape
    )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        f"random_forest_{split_name}.pkl"
    )

    print(
        "\nLoading model:",
        model_path
    )

    model = joblib.load(
        model_path
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # REGRESSION METRICS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    pearson, spearman = calculate_correlations(
        y_test,
        y_pred
    )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    (
        threshold,
        macro_f1,
        balanced_acc,
        auroc,
        auprc,
        mcc
    ) = calculate_classification_metrics(
        y_train,
        y_test,
        y_pred
    )


    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    (
        precision50,
        recall100,
        ndcg100,
        enrichment100
    ) = calculate_ranking_metrics(
        test_df,
        y_test,
        y_pred
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("\nRESULTS")

    print(
        f"MAE              : {mae:.4f}"
    )

    print(
        f"RMSE             : {rmse:.4f}"
    )

    print(
        f"R²               : {r2:.4f}"
    )

    print(
        f"Pearson          : {pearson:.4f}"
    )

    print(
        f"Spearman         : {spearman:.4f}"
    )

    print(
        f"Synergy threshold: {threshold:.4f}"
    )

    print(
        f"Macro-F1         : {macro_f1:.4f}"
    )

    print(
        f"Balanced Accuracy: {balanced_acc:.4f}"
    )

    print(
        f"AUROC            : {auroc:.4f}"
    )

    print(
        f"AUPRC            : {auprc:.4f}"
    )

    print(
        f"MCC              : {mcc:.4f}"
    )

    print(
        f"Precision@50     : {precision50:.4f}"
    )

    print(
        f"Recall@100       : {recall100:.4f}"
    )

    print(
        f"nDCG@100         : {ndcg100:.4f}"
    )

    print(
        f"Enrichment@100   : {enrichment100:.4f}"
    )


    # --------------------------------------------------------
    # CLEAN MEMORY
    # --------------------------------------------------------

    result = {
        "Split": split_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Pearson": pearson,
        "Spearman": spearman,
        "Synergy_Threshold": threshold,
        "Macro_F1": macro_f1,
        "Balanced_Accuracy": balanced_acc,
        "AUROC": auroc,
        "AUPRC": auprc,
        "MCC": mcc,
        "Precision_at_50": precision50,
        "Recall_at_100": recall100,
        "nDCG_at_100": ndcg100,
        "Enrichment_at_100": enrichment100
    }


    del train_df
    del test_df
    del X_test
    del y_test
    del y_pred
    del model

    gc.collect()

    return result


# ============================================================
# MAIN
# ============================================================

results = []

for split_name in SPLITS:

    try:

        result = evaluate_split(
            split_name
        )

        results.append(
            result
        )

    except Exception as e:

        print(
            f"\nERROR evaluating {split_name}:"
        )

        print(
            repr(e)
        )

        gc.collect()


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

output_path = os.path.join(
    RESULTS_DIR,
    "comprehensive_metrics.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


print("\n")
print("=" * 70)
print("COMPREHENSIVE METRICS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print(
    "\nSaved:",
    output_path
)
