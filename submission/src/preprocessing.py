"""
==========================================================
Discoverathon 2026
Challenge 1: Drug Combination Activity Prediction
==========================================================

Author : Alupana Balaji Reddy

Description:
This module preprocesses the raw NCI-ALMANAC dataset.

Responsibilities:
1. Load dataset
2. Explore dataset
3. Remove unwanted rows
4. Save processed dataset

==========================================================
"""
import pandas as pd
from pathlib import Path

# ==========================
# Project Paths
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "ComboDrugGrowth_Nov2017.csv"

PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "drug_combinations.csv"

def load_dataset(file_path):
    """
    Load the raw NCI-ALMANAC dataset.

    Parameters
    ----------
    file_path : Path

    Returns
    -------
    pandas.DataFrame
    """

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    df = pd.read_csv(file_path)

    print("Dataset loaded successfully.")

    return df

if __name__ == "__main__":
    df = load_dataset(RAW_DATA)
    dataset_summary(df)

def dataset_summary(df):
    """
    Display basic information about the dataset.

    Parameters
    ----------
    df : pandas.DataFrame        Input dataset

    Returns
    -------
    None
    """

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Number of Rows    : {df.shape[0]}")
    print(f"Number of Columns : {df.shape[1]}")

    print("\nColumn Names")

    for column in df.columns:
        print("-", column)
