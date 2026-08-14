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
    matthews_corrcoef,
)

from scipy.stats import pearsonr, spearmanr


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

FEATURES_DIR = os.path.join(
    PROJECT_ROOT,
    "features"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# SPLITS
# ============================================================

SPLITS = [
    "random",
    "cold_combination",
    "cold_cell_line",
    "cold_drug",
]


# ============================================================
# MODEL
# ============================================================

MODEL_PREFIX = "extra_trees_depth25"


# ============================================================
# COLUMNS TO REMOVE
# ============================================================

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
# DATA PREPARATION
# ============================================================

def prepare_data(
    df,
    feature_columns=None,
):
    """
    Prepare dataframe for prediction.

    The preprocessing follows the same general logic used
    during model training:

    1. Remove target/leakage columns.
    2. Remove metadata columns.
    3. One-hot encode categorical variables.
    4. Convert remaining values to numeric.
    5. Replace missing values with zero.
    6. Reorder columns to exactly match the trained model.
    """

    remove_columns = (
        LEAKAGE_COLUMNS
        + METADATA_COLUMNS
    )

    remove_columns = [
        column
        for column in remove_columns
        if column in df.columns
    ]

    X = df.drop(
        columns=remove_columns
    )

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "string"
            ]
        )
        .columns
        .tolist()
    )

    if categorical_columns:

        X = pd.get_dummies(
            X,
            columns=categorical_columns,
            drop_first=False,
            dtype="int8",
        )

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.fillna(0)

    # --------------------------------------------------------
    # EXACT FEATURE ORDER FROM TRAINED MODEL
    # --------------------------------------------------------

    if feature_columns is not None:

        X = X.reindex(
            columns=feature_columns,
            fill_value=0,
        )

    return X


# ============================================================
# CORRELATION METRICS
# ============================================================

def calculate_correlations(
    y_true,
    y_pred,
):
    """
    Calculate Pearson and Spearman correlations.
    """

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    # Pearson
    try:

        pearson = pearsonr(
            y_true,
            y_pred
        ).statistic

    except Exception:

        pearson = np.nan

    # Spearman
    try:

        spearman = spearmanr(
            y_true,
            y_pred
        ).statistic

    except Exception:

        spearman = np.nan

    return (
        pearson,
        spearman
    )


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def calculate_classification_metrics(
    y_train,
    y_test,
    y_pred,
):
    """
    Define synergy hits using the 90th percentile of the
    TRAINING SCORE distribution.

    The test set is therefore NOT used to determine its
    own classification threshold.
    """

    y_train = np.asarray(
        y_train,
        dtype=float
    )

    y_test = np.asarray(
        y_test,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    # --------------------------------------------------------
    # TRAINING-BASED THRESHOLD
    # --------------------------------------------------------

    threshold = np.quantile(
        y_train,
        0.90
    )

    # --------------------------------------------------------
    # TRUE / PREDICTED CLASSES
    # --------------------------------------------------------

    y_true_binary = (
        y_test >= threshold
    ).astype(int)

    y_pred_binary = (
        y_pred >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # MACRO F1
    # --------------------------------------------------------

    macro_f1 = f1_score(
        y_true_binary,
        y_pred_binary,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # BALANCED ACCURACY
    # --------------------------------------------------------

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true_binary,
            y_pred_binary,
        )
    )

    # --------------------------------------------------------
    # AUROC
    # --------------------------------------------------------

    try:

        auroc = roc_auc_score(
            y_true_binary,
            y_pred,
        )

    except ValueError:

        auroc = np.nan

    # --------------------------------------------------------
    # AUPRC
    # --------------------------------------------------------

    try:

        auprc = average_precision_score(
            y_true_binary,
            y_pred,
        )

    except ValueError:

        auprc = np.nan

    # --------------------------------------------------------
    # MCC
    # --------------------------------------------------------

    try:

        mcc = matthews_corrcoef(
            y_true_binary,
            y_pred_binary,
        )

    except ValueError:

        mcc = np.nan

    return (
        threshold,
        macro_f1,
        balanced_accuracy,
        auroc,
        auprc,
        mcc,
    )


# ============================================================
# DCG
# ============================================================

def dcg_at_k(
    relevances,
    k,
):
    """
    Calculate Discounted Cumulative Gain.
    """

    relevances = np.asarray(
        relevances[:k],
        dtype=float
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
        (
            (2 ** relevances) - 1
        )
        / discounts
    )


# ============================================================
# RANKING METRICS
# ============================================================

def calculate_ranking_metrics(
    df,
    y_pred,
):
    """
    Calculate ranking metrics within each CELLNAME.

    For each cell line:

        1. Rank predictions from highest to lowest.
        2. Define actual top 10% as relevant.
        3. Calculate:
           - Precision@50
           - Precision@100
           - Recall@100
           - nDCG@100
           - Enrichment@100

    Final values are the mean across cell lines.
    """

    work = df.copy()

    work["PREDICTED_SCORE"] = (
        np.asarray(
            y_pred,
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Check required column
    # --------------------------------------------------------

    if "CELLNAME" not in work.columns:

        raise ValueError(
            "CELLNAME column is required "
            "for ranking metrics."
        )

    if "SCORE" not in work.columns:

        raise ValueError(
            "SCORE column is required "
            "for ranking metrics."
        )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    precision50_values = []

    precision100_values = []

    recall100_values = []

    ndcg100_values = []

    enrichment100_values = []

    # --------------------------------------------------------
    # GROUP BY CELL LINE
    # --------------------------------------------------------

    for cell_name, group in work.groupby(
        "CELLNAME",
        sort=False,
    ):

        group = group[
            [
                "SCORE",
                "PREDICTED_SCORE"
            ]
        ].copy()

        group["SCORE"] = pd.to_numeric(
            group["SCORE"],
            errors="coerce"
        )

        group["PREDICTED_SCORE"] = pd.to_numeric(
            group["PREDICTED_SCORE"],
            errors="coerce"
        )

        group = group.dropna(
            subset=[
                "SCORE",
                "PREDICTED_SCORE"
            ]
        )

        n = len(group)

        if n == 0:

            continue

        # ----------------------------------------------------
        # ACTUAL TOP 10% = RELEVANT
        # ----------------------------------------------------

        n_relevant = max(
            1,
            int(
                np.ceil(
                    0.10 * n
                )
            )
        )

        actual_order = (
            group["SCORE"]
            .sort_values(
                ascending=False
            )
            .index
        )

        relevant_indices = set(
            actual_order[
                :n_relevant
            ]
        )

        group["RELEVANT"] = (
            group.index
            .isin(
                relevant_indices
            )
            .astype(int)
        )

        # ----------------------------------------------------
        # RANK BY PREDICTION
        # ----------------------------------------------------

        ranked = group.sort_values(
            "PREDICTED_SCORE",
            ascending=False
        )

        # ----------------------------------------------------
        # PRECISION @ 50
        # ----------------------------------------------------

        k50 = min(
            50,
            n
        )

        top50 = ranked.head(
            k50
        )

        precision50 = (
            top50["RELEVANT"].sum()
            / k50
        )

        precision50_values.append(
            precision50
        )

        # ----------------------------------------------------
        # PRECISION @ 100
        # ----------------------------------------------------

        k100 = min(
            100,
            n
        )

        top100 = ranked.head(
            k100
        )

        precision100 = (
            top100["RELEVANT"].sum()
            / k100
        )

        precision100_values.append(
            precision100
        )

        # ----------------------------------------------------
        # RECALL @ 100
        # ----------------------------------------------------

        relevant_found = (
            top100["RELEVANT"].sum()
        )

        recall100 = (
            relevant_found
            / n_relevant
        )

        recall100 = min(
            recall100,
            1.0
        )

        recall100_values.append(
            recall100
        )

        # ----------------------------------------------------
        # nDCG @ 100
        # ----------------------------------------------------

        predicted_relevances = (
            top100["RELEVANT"]
            .to_numpy()
        )

        dcg = dcg_at_k(
            predicted_relevances,
            k100
        )

        # Ideal ranking contains all relevant items first.
        ideal_relevances = np.concatenate(
            [
                np.ones(
                    min(
                        n_relevant,
                        k100
                    )
                ),
                np.zeros(
                    max(
                        0,
                        k100 - n_relevant
                    )
                ),
            ]
        )

        ideal_dcg = dcg_at_k(
            ideal_relevances,
            k100
        )

        if ideal_dcg > 0:

            ndcg100 = (
                dcg
                / ideal_dcg
            )

        else:

            ndcg100 = 0.0

        ndcg100_values.append(
            ndcg100
        )

        # ----------------------------------------------------
        # ENRICHMENT @ 100
        # ----------------------------------------------------

        prevalence = (
            n_relevant
            / n
        )

        if prevalence > 0:

            enrichment100 = (
                precision100
                / prevalence
            )

        else:

            enrichment100 = np.nan

        enrichment100_values.append(
            enrichment100
        )

    # ========================================================
    # MEAN ACROSS CELL LINES
    # ========================================================

    return (
        np.nanmean(
            precision50_values
        )
        if precision50_values
        else np.nan,

        np.nanmean(
            precision100_values
        )
        if precision100_values
        else np.nan,

        np.nanmean(
            recall100_values
        )
        if recall100_values
        else np.nan,

        np.nanmean(
            ndcg100_values
        )
        if ndcg100_values
        else np.nan,

        np.nanmean(
            enrichment100_values
        )
        if enrichment100_values
        else np.nan,
    )


# ============================================================
# EVALUATE ONE SPLIT
# ============================================================

def evaluate_split(
    split_name,
):
    """
    Evaluate Extra Trees depth-25 on one split.
    """

    print()
    print("=" * 80)
    print(
        f"EXTRA TREES DEPTH-25 — {split_name.upper()}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------------

    train_file = os.path.join(
        FEATURES_DIR,
        split_name,
        "train_features.csv"
    )

    test_file = os.path.join(
        FEATURES_DIR,
        split_name,
        "test_features.csv"
    )

    model_file = os.path.join(
        MODEL_DIR,
        f"{MODEL_PREFIX}_{split_name}.pkl"
    )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    for path in [
        train_file,
        test_file,
        model_file,
    ]:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"\nRequired file not found:\n{path}"
            )

    print(
        "\nModel:"
    )

    print(
        model_file
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print(
        "\nLoading model..."
    )

    model = joblib.load(
        model_file
    )

    print(
        "Model type:",
        type(model).__name__
    )

    print(
        "Trees:",
        model.n_estimators
    )

    print(
        "Max depth:",
        model.max_depth
    )

    print(
        "Min samples leaf:",
        model.min_samples_leaf
    )

    print(
        "Max features:",
        model.max_features
    )

    print(
        "Number of features:",
        len(
            model.feature_names_in_
        )
    )

    # --------------------------------------------------------
    # LOAD TRAINING DATA
    # --------------------------------------------------------

    print(
        "\nLoading training data..."
    )

    train_df = pd.read_csv(
        train_file,
        low_memory=False
    )

    print(
        "Training shape:",
        train_df.shape
    )

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    print(
        "\nLoading test data..."
    )

    test_df = pd.read_csv(
        test_file,
        low_memory=False
    )

    print(
        "Test shape:",
        test_df.shape
    )

    # --------------------------------------------------------
    # PREPARE TEST DATA
    # --------------------------------------------------------

    print(
        "\nPreparing test features..."
    )

    X_test = prepare_data(
        test_df,
        feature_columns=model.feature_names_in_
    )

    print(
        "Prediction matrix:",
        X_test.shape
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    y_test = pd.to_numeric(
        test_df["SCORE"],
        errors="coerce"
    )

    valid_mask = (
        y_test.notna()
    )

    y_test = (
        y_test[
            valid_mask
        ]
        .to_numpy(
            dtype=float
        )
    )

    X_test = X_test.loc[
        valid_mask
    ]

    test_eval_df = test_df.loc[
        valid_mask
    ].copy()

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print(
        "\nGenerating predictions..."
    )

    y_pred = model.predict(
        X_test
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    print(
        "Predictions generated:",
        len(y_pred)
    )

    # ========================================================
    # REGRESSION METRICS
    # ========================================================

    print(
        "\nCalculating regression metrics..."
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    pearson, spearman = (
        calculate_correlations(
            y_test,
            y_pred
        )
    )

    # ========================================================
    # TRAINING TARGET
    # ========================================================

    y_train = pd.to_numeric(
        train_df["SCORE"],
        errors="coerce"
    )

    y_train = (
        y_train
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    # ========================================================
    # CLASSIFICATION METRICS
    # ========================================================

    print(
        "Calculating classification metrics..."
    )

    (
        synergy_threshold,
        macro_f1,
        balanced_accuracy,
        auroc,
        auprc,
        mcc,
    ) = calculate_classification_metrics(
        y_train,
        y_test,
        y_pred,
    )

    # ========================================================
    # RANKING METRICS
    # ========================================================

    print(
        "Calculating ranking metrics..."
    )

    (
        precision50,
        precision100,
        recall100,
        ndcg100,
        enrichment100,
    ) = calculate_ranking_metrics(
        test_eval_df,
        y_pred,
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("-" * 80)
    print(
        "FINAL METRICS"
    )
    print("-" * 80)

    print(
        f"MAE              : {mae:.6f}"
    )

    print(
        f"RMSE             : {rmse:.6f}"
    )

    print(
        f"R2               : {r2:.6f}"
    )

    print(
        f"Pearson          : {pearson:.6f}"
    )

    print(
        f"Spearman         : {spearman:.6f}"
    )

    print(
        f"Synergy Threshold: {synergy_threshold:.6f}"
    )

    print(
        f"Macro F1         : {macro_f1:.6f}"
    )

    print(
        f"Balanced Accuracy: {balanced_accuracy:.6f}"
    )

    print(
        f"AUROC            : {auroc:.6f}"
    )

    print(
        f"AUPRC            : {auprc:.6f}"
    )

    print(
        f"MCC              : {mcc:.6f}"
    )

    print(
        f"Precision@50     : {precision50:.6f}"
    )

    print(
        f"Precision@100    : {precision100:.6f}"
    )

    print(
        f"Recall@100       : {recall100:.6f}"
    )

    print(
        f"nDCG@100         : {ndcg100:.6f}"
    )

    print(
        f"Enrichment@100   : {enrichment100:.6f}"
    )

    print("-" * 80)

    # ========================================================
    # RESULT DICTIONARY
    # ========================================================

    result = {
        "Split": split_name,

        "Model": MODEL_PREFIX,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2,

        "Pearson": pearson,

        "Spearman": spearman,

        "Synergy_Threshold": synergy_threshold,

        "Macro_F1": macro_f1,

        "Balanced_Accuracy": balanced_accuracy,

        "AUROC": auroc,

        "AUPRC": auprc,

        "MCC": mcc,

        "Precision_at_50": precision50,

        "Precision_at_100": precision100,

        "Recall_at_100": recall100,

        "nDCG_at_100": ndcg100,

        "Enrichment_at_100": enrichment100,
    }

    # --------------------------------------------------------
    # CLEAN MEMORY
    # --------------------------------------------------------

    del model
    del train_df
    del test_df
    del test_eval_df
    del X_test
    del y_test
    del y_train
    del y_pred

    gc.collect()

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "DISCOVERATHON 2026"
    )
    print(
        "COMPREHENSIVE EXTRA TREES DEPTH-25 EVALUATION"
    )
    print("=" * 80)

    print()
    print(
        "Model configuration:"
    )

    print(
        "ExtraTreesRegressor"
    )

    print(
        "n_estimators = 50"
    )

    print(
        "max_depth = 25"
    )

    print(
        "min_samples_leaf = 1"
    )

    print(
        "max_features = 1.0"
    )

    print(
        "random_state = 42"
    )

    print()
    print(
        "Splits:"
    )

    for split in SPLITS:

        print(
            f"  - {split}"
        )

    # --------------------------------------------------------
    # EVALUATE ALL SPLITS
    # --------------------------------------------------------

    all_results = []

    for split_name in SPLITS:

        result = evaluate_split(
            split_name
        )

        all_results.append(
            result
        )

    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    # --------------------------------------------------------
    # COLUMN ORDER
    # --------------------------------------------------------

    columns = [
        "Split",
        "Model",
        "MAE",
        "RMSE",
        "R2",
        "Pearson",
        "Spearman",
        "Synergy_Threshold",
        "Macro_F1",
        "Balanced_Accuracy",
        "AUROC",
        "AUPRC",
        "MCC",
        "Precision_at_50",
        "Precision_at_100",
        "Recall_at_100",
        "nDCG_at_100",
        "Enrichment_at_100",
    ]

    results_df = results_df[
        columns
    ]

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = os.path.join(
        RESULTS_DIR,
        "comprehensive_metrics.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 100)
    print(
        "FINAL COMPREHENSIVE METRICS"
    )
    print("=" * 100)

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "Saved:"
    )

    print(
        output_file
    )

    print(
        "=" * 100
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
