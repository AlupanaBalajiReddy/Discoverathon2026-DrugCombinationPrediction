# Drug Combination Synergy Prediction Using Machine Learning

## Discoverathon 2026 Submission

## 1. Project Overview

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

    data/processed/
    ├── drug_descriptors.csv
    ├── drug_smiles.csv
    └── unique_drugs.csv

Evaluation split files are provided under:

    splits/
    ├── random/
    ├── cold_combination/
    ├── cold_cell_line/
    └── cold_drug/

The target variable is the experimentally measured synergy score (`SCORE`).

## 7. Generalization Analysis

The random split was used as the reference for evaluating generalization.

| Split | R2 | RMSE | MAE |
|---|---:|---:|---:|
| Random | 0.4532 | 9.1981 | 6.2009 |
| Cold Combination | 0.1735 | 11.3680 | 7.1638 |
| Cold Cell Line | 0.3052 | 10.9512 | 6.8105 |
| Cold Drug | 0.0804 | 11.0555 | 7.3559 |

The results demonstrate reduced performance under cold-start conditions, particularly for previously unseen drugs.

## 8. Data-Efficiency Analysis

| Training Data | Samples | RMSE | MAE | R2 |
|---:|---:|---:|---:|---:|
| 1% | 22,971 | 12.1800 | 7.7750 | 0.0413 |
| 5% | 114,857 | 11.1691 | 7.2126 | 0.1938 |
| 10% | 229,715 | 10.6704 | 6.9535 | 0.2642 |
| 25% | 574,288 | 10.0917 | 6.6511 | 0.3419 |
| 100% | 2,297,155 | 9.1981 | 6.2009 | 0.4532 |

Performance improves as the amount of training data increases.

## 9. Prediction

The final random-split Extra Trees model can be used through `src/predict.py`.

Example:

    python src/predict.py input.csv predictions.csv

The pipeline loads the final checkpoint, removes leakage and metadata columns, encodes categorical variables, aligns the feature order, and generates the `PREDICTED_SCORE` column.

## 10. Reproducibility

This section provides the exact instructions required to reproduce the project on an independent Linux or Windows system.

The repository contains the source code, processed molecular data, evaluation splits, trained model checkpoints, results and figures required for reproduction.

There are two reproduction modes:

### A. Reproduce the submitted experiment

This is the recommended approach for verifying the reported results.

Use the supplied:

- Processed molecular descriptor files
- Train/validation/test split files
- Trained model checkpoints
- Source scripts
- Evaluation result files

The supplied split files should be used when reproducing the reported results because they preserve the exact train/validation/test assignments used during the experiments.

### B. Reproduce the complete pipeline from the raw dataset

The complete pipeline can be regenerated from the NCI-ALMANAC raw dataset by following the preprocessing, split-generation, feature-generation, training, prediction and evaluation workflow described below.

---

### 10.1 Required Software

The project was developed and tested using:

```text
Python 3.11.15
pandas==3.0.5
numpy==2.4.6
scikit-learn==1.9.0
rdkit==2026.03.5
xgboost==3.2.0
lightgbm==4.7.0
catboost==1.2.10
shap==0.51.0
joblib==1.5.3
matplotlib==3.11.1
scipy==1.17.1


## 11. Model Selection

The depth-25 Extra Trees model was selected as a practical trade-off between predictive performance and checkpoint size. The unrestricted-depth model produced stronger random-split performance but substantially larger checkpoints.

An ANN experiment using a 500,000-sample training subset achieved a test R2 of approximately 0.160 and was not selected as the final model.

## 12. AI Assistance and Acknowledgement

AI tools, including ChatGPT, were used during development for code assistance, debugging, documentation, presentation preparation, and clarification of technical concepts.

All data preprocessing, experimental design, model training, evaluation, interpretation of results, and final scientific conclusions were reviewed and validated by the project team.

AI tools were not used as a zero-shot prediction system for the challenge. The submitted predictions were generated using task-specific machine-learning models trained on the challenge data.
