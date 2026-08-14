# Predicting Drug Combination Activity

## Discoverathon 2026 Submission

### 1. Project Overview

This project addresses the problem of **Predicting Drug Combination Activity** using machine learning.

The objective is to predict drug-combination synergy/activity scores using molecular descriptors, drug-pair information, and experimental features derived from the NCI-ALMANAC dataset.

Multiple machine-learning approaches were evaluated, including:

- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost
- HistGradientBoosting
- Random Forest + Extra Trees ensemble
- Artificial Neural Network (ANN)

The final selected model is an **Extra Trees Regressor** with controlled tree depth.

---


## 2. Final Models

Extra Trees Regressors with controlled tree depth were used for the four evaluation settings.

Configuration:

- Number of trees: 50
- Maximum depth: 25
- Minimum samples per leaf: 1
- Maximum features: 1.0
- Random state: 42
- Parallel processing: enabled

Checkpoints:

- `extra_trees_depth25_random.pkl`
- `extra_trees_depth25_cold_combination.pkl`
- `extra_trees_depth25_cold_cell_line.pkl`
- `extra_trees_depth25_cold_drug.pkl`

The random-split model achieved the strongest overall performance:

- RMSE: 9.1981
- MAE: 6.2009
- R²: 0.4532

The cold-start models are provided to reproduce evaluation on unseen drug combinations, cell lines, and drugs.

## 3. Evaluation Protocols

The model was evaluated using four data-splitting strategies:

1. Random split
2. Cold-combination split
3. Cold-cell-line split
4. Cold-drug split

These evaluations measure both conventional predictive performance and generalization to previously unseen combinations, cell lines, and drugs.

---

## 4. Final Test Results

| Split | RMSE | MAE | R² |
|---|---:|---:|---:|
| Random | 9.1981 | 6.2009 | 0.4532 |
| Cold Combination | 11.3680 | 7.1638 | 0.1735 |
| Cold Cell Line | 10.9512 | 6.8105 | 0.3052 |
| Cold Drug | 11.0555 | 7.3559 | 0.0804 |

### Best random-split performance

- RMSE: 9.1981
- MAE: 6.2009
- R²: 0.4532

The cold-start experiments demonstrate that prediction for unseen drugs is substantially more challenging than prediction for unseen cell lines or previously observed drug combinations.

---

## 5. Model Benchmarking

Several models were evaluated during model development.

The final Extra Trees depth-25 model achieved substantially better random-split performance than the earlier baseline and ensemble approaches.

An ANN experiment was also conducted using a 500,000-sample training subset. Its test R² was approximately 0.160 and therefore it was not selected as the final model.

The unrestricted-depth Extra Trees model produced higher random-split performance, but generated substantially larger model checkpoints. The depth-25 configuration provided a better practical trade-off between predictive performance and model size.

---

## 6. Data

Processed molecular information is provided under:

```text
data/processed/
The processed data includes:

- `drug_descriptors.csv`
- `drug_smiles.csv`
- `unique_drugs.csv`

Evaluation split files are provided under:

```text
splits/
