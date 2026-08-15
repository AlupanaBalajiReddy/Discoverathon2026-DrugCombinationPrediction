# Predicting Drug Combination Activity

**Discoverathon 2026**

**Assigned Problem Statement:** Predicting Drug Combination Activity

---

# 1. Project Overview

Drug combination therapy is an important strategy in cancer research because combining therapeutic agents can produce enhanced, additive, or synergistic effects. However, experimentally evaluating a large number of drug combinations across multiple cancer cell lines is expensive and time-consuming.

This project develops a machine-learning pipeline for **predicting anticancer drug-combination activity/synergy scores** using the **NCI-ALMANAC** dataset.

The workflow integrates:

- Drug molecular descriptors
- Drug-pair information
- Drug concentration information
- Cancer cell-line information
- Cancer-panel information

The project evaluates conventional random-split performance as well as generalization to previously unseen:

- Drug combinations
- Cancer cell lines
- Drugs

The central objective is to determine how effectively machine-learning models can predict drug-combination activity and how prediction performance changes when the model encounters entities not represented in the training data.

---

# 2. Problem Statement

## Predicting Drug Combination Activity

Experimental screening of large numbers of drug combinations across cancer cell lines is costly and time-consuming.

Machine learning can learn relationships between experimentally measured drug-combination responses and molecular, experimental, and biological features. Such models can potentially help prioritize combinations for further investigation.

The objective of this project is to predict the experimentally measured **`SCORE`** and evaluate predictive performance under both conventional random-split and realistic cold-start conditions.

---

# 3. Objectives

The main objectives are:

1. Explore and preprocess the NCI-ALMANAC dataset.
2. Obtain molecular information for the drugs.
3. Generate molecular descriptors using RDKit.
4. Construct drug-pair representations.
5. Integrate molecular, experimental, and biological features.
6. Avoid use of target-derived information during prediction.
7. Generate reproducible train/validation/test splits.
8. Evaluate conventional random-split performance.
9. Evaluate cold-combination generalization.
10. Evaluate cold-cell-line generalization.
11. Evaluate cold-drug generalization.
12. Compare machine-learning models.
13. Analyze model generalization.
14. Analyze the effect of training-data size.
15. Investigate feature importance and SHAP-based explainability.
16. Provide a reproducible prediction pipeline.

---

# 4. Dataset

## 4.1 NCI-ALMANAC

The project uses the **NCI-ALMANAC drug-combination dataset** from the National Cancer Institute.

The processed dataset used for model development contains:

| Property | Value |
|---|---:|
| Valid experimental observations | 2,871,444 |
| Unique drugs | 105 |
| Unique drug pairs | 5,242 |
| Cancer cell lines | 60 |

The original dataset contained 3,686,475 experimental rows before removal of observations without the required synergy/activity target.

---

## 4.2 Prediction Target

The prediction target is:

```text
SCORE
```

`SCORE` is the experimentally measured drug-combination activity/synergy value used as the regression target.

The primary machine-learning task is therefore formulated as a **supervised regression problem**.

---

# 5. Molecular Representation

Drug molecular structures were processed using **RDKit**.

The molecular descriptor set contains the following descriptors for each drug:

| Descriptor |
|---|
| MolWt |
| ExactMolWt |
| MolLogP |
| TPSA |
| NumHDonors |
| NumHAcceptors |
| NumRotatableBonds |
| HeavyAtomCount |
| RingCount |
| NumAromaticRings |
| FractionCSP3 |
| MolMR |

These descriptors are generated separately for Drug 1 and Drug 2.

Therefore, the molecular representation contributes:

\[
12 \times 2 = 24
\]

descriptor features.

The processed molecular information included in the submission is:

```text
submission/data/processed/
├── drug_descriptors.csv
├── drug_smiles.csv
└── unique_drugs.csv
```

---

# 6. Feature Representation

The final random-split Extra Trees model uses **107 input features**.

These features are not all molecular descriptors. They combine molecular, experimental, concentration, validation, panel, and cell-line information.

## 6.1 Experimental and Drug-Pair Features

The first 11 features are:

```text
COMBODRUGSEQ
PANELNBR
CELLNBR
NSC1
SAMPLE1
CONCINDEX1
CONC1
NSC2
SAMPLE2
CONCINDEX2
CONC2
```

---

## 6.2 Drug 1 Molecular Features

Drug 1 contributes:

```text
MolWt_1
ExactMolWt_1
MolLogP_1
TPSA_1
NumHDonors_1
NumHAcceptors_1
NumRotatableBonds_1
HeavyAtomCount_1
RingCount_1
NumAromaticRings_1
FractionCSP3_1
MolMR_1
```

---

## 6.3 Drug 2 Molecular Features

Drug 2 contributes:

```text
MolWt_2
ExactMolWt_2
MolLogP_2
TPSA_2
NumHDonors_2
NumHAcceptors_2
NumRotatableBonds_2
HeavyAtomCount_2
RingCount_2
NumAromaticRings_2
FractionCSP3_2
MolMR_2
```

Thus, the two drugs contribute:

\[
12 + 12 = 24
\]

RDKit molecular descriptors.

---

## 6.4 Additional Numerical Features

The feature representation also includes:

```text
CONCUNIT1_M
CONCUNIT2_M
VALID_Y
```

---

## 6.5 Cancer-Panel Features

Panel information is represented using one-hot encoded features for the observed cancer panels.

The final model contains nine panel indicator features:

```text
PANEL_Breast Cancer
PANEL_CNS Cancer
PANEL_Colon Cancer
PANEL_Leukemia
PANEL_Melanoma
PANEL_Non-Small Cell Lung Cancer
PANEL_Ovarian Cancer
PANEL_Prostate Cancer
PANEL_Renal Cancer
```

---

## 6.6 Cancer Cell-Line Features

Cancer cell-line information is represented using one-hot encoded cell-line features.

The random-split model contains 58 cell-line indicator features.

The verified final random-split checkpoint contains **107 model input features**.

These features consist of:

- Experimental and drug-pair variables
- Molecular descriptors for Drug 1
- Molecular descriptors for Drug 2
- Concentration-related variables
- Validation information
- Cancer-panel encoding
- Cancer-cell-line encoding

The exact feature names and ordering are stored in the trained model checkpoint and can be inspected using:

```python
model.feature_names_in_

```

The verified random-split checkpoint reports:

```text
Number of model features: 107
```

---

# 7. Target and Leakage Handling

The prediction pipeline removes target-related fields before generating model inputs.

The following target or response-derived fields are excluded from the prediction feature matrix:

```text
SCORE
PERCENTGROWTH
PERCENTGROWTHNOTZ
TESTVALUE
CONTROLVALUE
TZVALUE
EXPECTEDGROWTH
```

Additional metadata fields that are not used as predictive inputs include:

```text
SCREENER
STUDY
TESTDATE
PLATE
PREFIX1
PREFIX2
drug_pair_id
```

This prevents the prediction pipeline from directly passing the target or specified response-derived variables to the model.

The prediction script verifies the final feature matrix against the trained model's expected feature names before prediction.

---

# 8. Data Splitting Strategy

Four evaluation settings were considered.

```text
1. Random
2. Cold Combination
3. Cold Cell Line
4. Cold Drug
```

The split files are provided in:

```text
submission/splits/
```

---

## 8.1 Random Split

The random split provides the conventional baseline evaluation.

The data are divided into training, validation, and test sets.

```text
submission/splits/random/
├── train.csv
├── validation.csv
└── test.csv
```

The implemented random split uses an approximately:

```text
80% training
10% validation
10% test
```

partition.

---

## 8.2 Cold-Combination Split

The cold-combination setting evaluates generalization to drug combinations that are held out from model training.

```text
submission/splits/cold_combination/
├── train.csv
├── validation.csv
└── test.csv
```

This evaluates whether the model can generalize to previously unseen drug-pair combinations.

---

## 8.3 Cold-Cell-Line Split

The cold-cell-line setting evaluates generalization to held-out cancer cell lines.

```text
submission/splits/cold_cell_line/
├── train.csv
├── validation.csv
└── test.csv
```

This evaluates whether the model can generalize to biological contexts that were not represented in its training observations.

---

## 8.4 Cold-Drug Split

The cold-drug setting evaluates generalization to held-out drugs.

```text
submission/splits/cold_drug/
├── train.csv
├── validation.csv
├── test.csv
└── drug_assignment.csv
```

The drug assignment file records the drug-group allocation used for the cold-drug evaluation.

The cold-drug evaluation is particularly challenging because the model must predict combinations involving drugs excluded from the training data.

---

# 9. Machine-Learning Models

Several tree-based regression models were evaluated during model development:

- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost

The initial random-split comparison was used to assess the relative performance of these algorithms.

The final evaluation pipeline uses the **Extra Trees Regressor** model.

---

# 10. Final Extra Trees Models

The final checkpoints are:

```text
submission/checkpoints/
├── extra_trees_depth25_random.pkl
├── extra_trees_depth25_cold_combination.pkl
├── extra_trees_depth25_cold_cell_line.pkl
└── extra_trees_depth25_cold_drug.pkl
```

All four checkpoints were verified as:

```text
Model: ExtraTreesRegressor
Number of estimators: 50
Maximum depth: 25
Random state: 42
```

The feature counts are:

| Evaluation | Features | Estimators | Max Depth | Random State |
|---|---:|---:|---:|---:|
| Random | 107 | 50 | 25 | 42 |
| Cold Combination | 107 | 50 | 25 | 42 |
| Cold Cell Line | 95 | 50 | 25 | 42 |
| Cold Drug | 107 | 50 | 25 | 42 |

The difference in feature count for the cold-cell-line model results from the entity-specific encoded feature representation under the held-out cell-line setting.

---

# 11. Model Evaluation

The project evaluates the models using regression, correlation, classification, and ranking metrics.

---

## 11.1 Regression Metrics

### Mean Absolute Error

\[
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
\]

### Root Mean Squared Error

\[
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
\]

### Coefficient of Determination

\[
R^2 =
1 -
\frac{
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
}{
\sum_{i=1}^{n}(y_i-\bar{y})^2
}
\]

where:

- $y_i$ = observed score
- $\hat{y}_i$ = predicted score
- $\bar{y}$ = mean observed score
- $n$ = number of observations

---

## 11.2 Correlation Metrics

The model is additionally evaluated using:

- Pearson correlation
- Spearman rank correlation

Pearson correlation measures linear association between observed and predicted values, while Spearman correlation evaluates monotonic rank association.

---

## 11.3 Classification Metrics

Predictions are additionally evaluated for identifying high-synergy observations using:

```text
SCORE >= 8.0
```

The classification metrics include:

- Macro F1
- Balanced Accuracy
- AUROC
- AUPRC
- Matthews Correlation Coefficient (MCC)

---

## 11.4 Ranking Metrics

The ability to prioritize high-scoring combinations is evaluated using:

- Precision@50
- Precision@100
- Recall@100
- nDCG@100
- Enrichment@100

These metrics evaluate whether high-synergy combinations are concentrated near the top of the prediction ranking.

---

# 12. Initial Model Comparison

The initial random-split model comparison produced:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Random Forest | 10.3318 | 6.8519 | 0.3102 |
| Extra Trees | 10.4061 | 6.8709 | 0.3002 |
| XGBoost | 10.7718 | 7.0036 | 0.2502 |
| LightGBM | 11.2041 | 7.2471 | 0.1888 |
| CatBoost | 11.4430 | 7.3145 | 0.1538 |

The final Extra Trees configuration was subsequently evaluated using the comprehensive evaluation pipeline.

---

# 13. Final Random-Split Performance

The final `extra_trees_depth25` model achieved:

| Metric | Random Split |
|---|---:|
| MAE | 6.2009 |
| RMSE | 9.1981 |
| R² | 0.4532 |
| Pearson | 0.6770 |
| Spearman | 0.5395 |
| Macro F1 | 0.6455 |
| Balanced Accuracy | 0.6075 |
| AUROC | 0.7970 |
| AUPRC | 0.4725 |
| MCC | 0.3686 |
| Precision@50 | 0.8390 |
| Precision@100 | 0.7825 |
| Recall@100 | 0.1639 |
| nDCG@100 | 0.8120 |
| Enrichment@100 | 7.8173 |

---

# 14. Cold-Start Generalization

The final Extra Trees model was evaluated under three cold-start settings.

| Evaluation | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random | 6.2009 | 9.1981 | 0.4532 |
| Cold Combination | 7.1638 | 11.3680 | 0.1735 |
| Cold Cell Line | 6.8105 | 10.9512 | 0.3052 |
| Cold Drug | 7.3559 | 11.0555 | 0.0804 |

The results demonstrate a reduction in predictive performance when the model is evaluated on entities that were not represented in training.

The strongest degradation in R² occurs under the cold-drug setting.

---

# 15. Generalization Gap

The generalization gap was calculated relative to the random-split result.

| Evaluation | R² Gap | RMSE Change | MAE Change |
|---|---:|---:|---:|
| Random | 0.0000 | 0.0000 | 0.0000 |
| Cold Combination | -0.2798 | +2.1699 | +0.9629 |
| Cold Cell Line | -0.1480 | +1.7532 | +0.6096 |
| Cold Drug | -0.3729 | +1.8574 | +1.1550 |

The cold-drug setting has the largest R² degradation relative to the random baseline.

---

# 16. Data-Efficiency Analysis

The effect of training-data size was evaluated using the final Extra Trees approach.

| Training Data | Samples | RMSE | MAE | R² |
|---:|---:|---:|---:|---:|
| 1% | 22,971 | 12.1800 | 7.7750 | 0.0413 |
| 5% | 114,857 | 11.1691 | 7.2126 | 0.1938 |
| 10% | 229,715 | 10.6704 | 6.9535 | 0.2642 |
| 25% | 574,288 | 10.0917 | 6.6511 | 0.3419 |
| 100% | 2,297,155 | 9.1981 | 6.2009 | 0.4532 |

Increasing the amount of training data improves predictive performance.

R² increases from approximately 0.04 with 1% of the training data to approximately 0.45 with the full training set.

---

# 17. Model Explainability

Feature-importance analysis and SHAP-based analyses were performed to investigate model behavior.

The submission contains:

```text
submission/figures/
├── feature_importance.png
├── shap_feature_importance.png
└── shap_summary.png
```

These analyses provide information about the relative contribution of model features to prediction behavior.

---

# 18. Ensemble Analysis

Ensemble-based evaluation was also performed.

Relevant files include:

```text
submission/results/
├── ensemble_comparison.csv
├── ensemble_validation.csv
├── ensemble_weights.csv
└── cold_ensemble_comparison.csv
```

Cold-start ensemble comparisons are provided separately from the primary Extra Trees model results.

---

# 19. Prediction Pipeline

The prediction interface is:

```bash
python submission/src/predict.py <input.csv> <output.csv>
```

Example:

```bash
python submission/src/predict.py \
submission/splits/random/test.csv \
submission/predictions.csv
```

The prediction pipeline:

1. Loads the final random-split Extra Trees checkpoint.
2. Reads the input CSV.
3. Removes target and excluded metadata fields.
4. Generates the required molecular features.
5. Constructs the model input matrix.
6. Aligns the feature matrix with the trained model's expected feature names.
7. Generates predictions.
8. Writes the predictions to the requested output file.

The final model contains 107 expected input features.

---

# 20. Prediction Verification

The prediction pipeline was tested using:

```text
submission/splits/random/test.csv
```

The test input contained:

```text
287,145 observations
```

The prediction matrix contained:

```text
287,145 × 107 features
```

Predictions were successfully generated for all:

```text
287,145 observations
```

The verified prediction output contained:

```text
PREDICTED_SCORE
```

for every test observation.

The final submission prediction file is:

```text
submission/test_predictions.csv
```

---

# 21. Repository Structure

```text
Discoverathon2026/
├── README.md
├── PROJECT_LOG.md
├── requirements.txt
├── backup/
├── catboost_info/
├── checkpoints/
├── data/
├── features/
├── figures/
├── models/
├── notebooks/
├── presentation/
├── reports/
├── results/
├── splits/
├── src/
└── submission/
    ├── README.md
    ├── requirements.txt
    ├── checkpoints/
    ├── data/
    │   └── processed/
    ├── figures/
    ├── results/
    ├── splits/
    ├── src/
    └── test_predictions.csv
```

---

# 22. Submission Directory

The `submission/` directory contains the main reproducibility artifacts.

## Source Code

```text
submission/src/
```

## Processed Molecular Data

```text
submission/data/processed/
```

## Model Checkpoints

```text
submission/checkpoints/
```

## Evaluation Results

```text
submission/results/
```

## Figures

```text
submission/figures/
```

## Data Splits

```text
submission/splits/
```

## Predictions

```text
submission/test_predictions.csv
```

---

# 23. Submission Results

Important result files include:

```text
submission/results/
├── comprehensive_metrics.csv
├── data_efficiency.csv
├── ensemble_comparison.csv
├── ensemble_validation.csv
├── ensemble_weights.csv
├── extra_trees_cold_splits.csv
├── extra_trees_depth25_cold_splits.csv
├── extra_trees_depth25_random.csv
├── final_model_results.csv
├── generalization_gap.csv
├── model_comparison_random_final.csv
└── random_forest_all_splits.csv
```

These files contain the numerical results used for model comparison and analysis.

---

# 24. Figures

The submission contains the following principal visualizations:

```text
submission/figures/
├── cold_start_r2.png
├── cold_start_rmse.png
├── data_efficiency_r2.png
├── data_efficiency_rmse.png
├── feature_importance.png
├── final_model_comparison.png
├── final_r2_comparison.png
├── generalization_r2.png
├── generalization_rmse.png
├── shap_feature_importance.png
└── shap_summary.png
```

---

# 25. Available Analysis Scripts

The submission contains:

```text
submission/src/
├── preprocessing.py
├── generate_features.py
├── generate_splits.py
├── train_all_splits.py
├── evaluate_all_metrics.py
├── evaluate_ensemble.py
├── evaluate_cold_ensemble.py
├── analyze_generalization.py
├── data_efficiency.py
├── plot_data_efficiency.py
└── predict.py
```

| Script | Purpose |
|---|---|
| `preprocessing.py` | Data preprocessing |
| `generate_features.py` | Feature generation |
| `generate_splits.py` | Generation of evaluation splits |
| `train_all_splits.py` | Model training workflow |
| `evaluate_all_metrics.py` | Comprehensive evaluation |
| `evaluate_ensemble.py` | Ensemble evaluation |
| `evaluate_cold_ensemble.py` | Cold-start ensemble evaluation |
| `analyze_generalization.py` | Generalization-gap analysis |
| `data_efficiency.py` | Training-data efficiency analysis |
| `plot_data_efficiency.py` | Data-efficiency visualization |
| `predict.py` | Prediction on input data |

---

# 26. Software Requirements

The project uses Python and the dependencies specified in:

```text
requirements.txt
submission/requirements.txt
```

The required versions include:

```text
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
```

---

# 27. Reproducibility

The project is designed so that the code, model checkpoints, processed molecular data, evaluation splits, results, figures, and prediction output can be inspected independently.

The repository contains the source code and smaller submission artifacts, while large artifacts are additionally maintained through Git LFS and Zenodo.

---

# 28. Git LFS

Large repository artifacts are tracked using **Git Large File Storage (Git LFS)**.

The repository currently contains **18 Git LFS-tracked files**.

These include:

- Four Extra Trees model checkpoints
- Random train/validation/test splits
- Cold-combination train/validation/test splits
- Cold-cell-line train/validation/test splits
- Cold-drug train/validation/test splits
- Cold-drug assignment information
- Test predictions

LFS integrity was verified using:

```bash
git lfs fsck
```

with the result:

```text
Git LFS fsck OK
```

---

# 29. Large Artifact Archive

Because several trained models and dataset splits are large, the large reproducibility artifacts were packaged separately.

The archive contains:

```text
checkpoints/
splits/
test_predictions.csv
```

The archive contains **18 files**.

The prepared archive was:

```text
Discoverathon2026_large_artifacts.tar.gz
```

The archive was validated successfully using:

```bash
tar -tzf Discoverathon2026_large_artifacts.tar.gz
```

---

# 30. Zenodo Reproducibility Record

The large-artifact archive is also available through Zenodo.

**DOI:**

```text
10.5281/zenodo.21945858
```

**Record:**

https://doi.org/10.5281/zenodo.21945858

The Zenodo record provides access to the large reproducibility artifacts that are not convenient to distribute as ordinary repository files.

The archive includes:

- Trained model checkpoints
- Random evaluation splits
- Cold-combination evaluation splits
- Cold-cell-line evaluation splits
- Cold-drug evaluation splits
- Cold-drug assignment information
- Final test predictions

---

# 31. Archive Integrity

The SHA256 checksum of the prepared large-artifact archive is:

```text
d506a3b1da38a792145c299eff1fbf70d2a8f6c17a7deb77219cb608c7c0ed37
```

The archive was validated before upload.

The archive contains:

```text
18 files
```

The checksum can be independently verified using:

```bash
sha256sum Discoverathon2026_large_artifacts.tar.gz
```

---

# 32. Reproduction Workflow

## 32.1 Clone the Repository

Using SSH:

```bash
git clone git@github.com:AlupanaBalajiReddy/Discoverathon2026-DrugCombinationPrediction.git
cd Discoverathon2026-DrugCombinationPrediction
```

Using HTTPS:

```bash
git clone https://github.com/AlupanaBalajiReddy/Discoverathon2026-DrugCombinationPrediction.git
cd Discoverathon2026-DrugCombinationPrediction
```

---

## 32.2 Initialize Git LFS

```bash
git lfs install
git lfs pull
```

---

## 32.3 Create the Python Environment

```bash
conda create -n discoverathon python=3.11 -y
conda activate discoverathon
```

---

## 32.4 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 32.5 Generate Predictions

```bash
python submission/src/predict.py \
submission/splits/random/test.csv \
submission/predictions.csv
```

The output file will contain the original input columns together with:

```text
PREDICTED_SCORE
```

---

# 33. Interpretation of Results

The results demonstrate that the model learns meaningful relationships between the input features and experimentally measured drug-combination scores.

Performance is strongest under the conventional random split.

Performance decreases under cold-start settings because the model must generalize to entities or combinations that were not represented in the training data.

The cold-drug setting is particularly challenging.

The data-efficiency analysis also demonstrates that increasing training-data availability improves model performance.

---

# 34. Key Findings

The major findings are:

1. The final Extra Trees model achieved an R² of approximately **0.45** under the random split.
2. The cold-combination evaluation achieved an R² of approximately **0.17**.
3. The cold-cell-line evaluation achieved an R² of approximately **0.31**.
4. The cold-drug evaluation achieved an R² of approximately **0.08**.
5. Cold-drug evaluation produced the largest R² degradation relative to the random split.
6. The model achieved an AUROC of approximately **0.80** under the random split for identifying high-synergy observations using the specified threshold.
7. Precision@100 under the random split was approximately **0.78**.
8. Training-data efficiency analysis showed consistent improvement as more training data were used.
9. SHAP and feature-importance analyses provide additional interpretability of the trained model.

---

# 35. Limitations

The following limitations should be considered:

- The dataset represents experimentally measured drug-combination responses within a defined screening system.
- The dataset may not represent every cancer type or clinical setting.
- Generalization to completely unseen drugs remains challenging.
- Molecular descriptors capture chemical properties but do not fully represent biological mechanisms of drug action.
- Cancer cell-line experiments do not completely reproduce patient-level tumor heterogeneity.
- Model predictions should be regarded as computational prioritization rather than substitutes for experimental validation.
- Performance depends on the distribution and coverage of the available training data.
- Cold-start performance is lower than random-split performance, demonstrating the difficulty of extrapolating beyond the observed data distribution.

---

# 36. Scientific Significance

The project evaluates model performance beyond conventional random splitting by explicitly testing generalization to:

- Unseen drug combinations
- Unseen cancer cell lines
- Unseen drugs

This provides a more demanding assessment of model generalization.

The difference between random and cold-start performance demonstrates that high performance on randomly sampled observations does not necessarily imply equivalent performance on previously unseen biological or chemical entities.

The results therefore highlight the importance of:

- Leakage-aware feature construction
- Entity-aware evaluation
- Cold-start testing
- Molecular representation
- Sufficient training-data coverage
- Model interpretability

for machine-learning approaches to drug-combination activity prediction.

---

# 37. Team

| Name | Role |
|---|---|
| Priyanka Yadav | Mentor |
| Alupana Balaji Reddy | Team Lead |
| Harish P S | Team Member |
| Sudhish Gupta | Team Member |

---

# 38. AI Assistance and Acknowledgement

AI tools, including ChatGPT, were used during the development of this project for code assistance, debugging, documentation, presentation preparation, and clarification of technical concepts.

All data preprocessing, experimental design, model training, evaluation, interpretation of results, and final scientific conclusions were reviewed and validated by the project team.

AI tools were not used as a zero-shot prediction system for the challenge. The submitted predictions were generated using task-specific machine-learning models trained on the challenge data.

---

# 39. Acknowledgement

The team acknowledges the National Cancer Institute for the NCI-ALMANAC resource used in this project.

The team also acknowledges the organizers of **Discoverathon 2026** for providing the problem statement and challenge framework.

---

# 40. Final Submission Checklist

The following components have been prepared:

- [x] Discoverathon problem statement documented
- [x] Project overview documented
- [x] Dataset documented
- [x] Prediction target documented
- [x] Molecular representation documented
- [x] Feature representation documented
- [x] Leakage handling documented
- [x] Random split documented
- [x] Cold-combination split documented
- [x] Cold-cell-line split documented
- [x] Cold-drug split documented
- [x] Machine-learning models documented
- [x] Final Extra Trees configuration documented
- [x] Regression metrics documented
- [x] Classification metrics documented
- [x] Ranking metrics documented
- [x] Final model results documented
- [x] Cold-start results documented
- [x] Generalization analysis documented
- [x] Data-efficiency analysis documented
- [x] Explainability documented
- [x] Prediction workflow documented
- [x] Repository structure documented
- [x] Submission artifacts documented
- [x] Software requirements documented
- [x] Git LFS documented
- [x] Large-artifact archive documented
- [x] Zenodo record documented
- [x] Archive checksum documented
- [x] Reproducibility instructions documented
- [x] Team documented
- [x] AI assistance acknowledged

---

# 41. External Resources

## GitHub Repository

https://github.com/AlupanaBalajiReddy/Discoverathon2026-DrugCombinationPrediction

## Zenodo Reproducibility Record

https://doi.org/10.5281/zenodo.21945858

---

**Discoverathon 2026 — Predicting Drug Combination Activity**
