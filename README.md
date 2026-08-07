# Drug Combination Synergy Prediction using Machine Learning

## Project Overview

Drug combination therapy has become an important strategy in cancer treatment because combining multiple drugs can improve therapeutic efficacy while reducing toxicity and drug resistance. However, experimentally evaluating all possible drug combinations is expensive and time-consuming.

This project develops an end-to-end machine learning pipeline to predict **drug combination synergy scores** using molecular descriptors generated from chemical structures. The workflow includes data preprocessing, feature engineering using RDKit, model development, model comparison, and performance evaluation.

---

## Problem Statement

The objective of this project is to predict the synergy score of drug combinations using molecular descriptors derived from the chemical structures of two drugs.

Accurate prediction of synergistic drug combinations can reduce experimental costs and accelerate drug discovery by prioritizing promising drug pairs for biological validation.

---

## Dataset

**Dataset:** NCI-ALMANAC Drug Combination Dataset

The dataset contains experimentally measured responses of anticancer drug combinations tested across multiple cancer cell lines.

### Target Variable

- Drug Combination Synergy Score

---

## Repository Structure

```text
Discoverathon2026-DrugCombinationPrediction/

├── data/
│   ├── raw/
│   ├── external/
│   └── processed/
│
├── notebooks/
│   ├── 00_project_setup.ipynb
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_final_analysis.ipynb
│
├── src/
│   └── preprocessing.py
│
├── results/
│   ├── model_comparison.csv
│   └── random_forest_feature_importance.csv
│
├── models/
│
├── README.md
├── PROJECT_LOG.md
├── requirements.txt
└── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/AlupanaBalajiReddy/Discoverathon2026-DrugCombinationPrediction.git

cd Discoverathon2026-DrugCombinationPrediction
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

## Project Workflow

```text
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
RDKit Molecular Descriptors
      │
      ▼
Train-Test Split
      │
      ▼
Baseline Random Forest
      │
      ▼
Model Comparison
      │
      ▼
Feature Importance Analysis
      │
      ▼
Final Analysis
```

---

## Running the Project

Execute the notebooks in the following order:

1. 00_project_setup.ipynb
2. 01_dataset_exploration.ipynb
3. 02_data_preprocessing.ipynb
4. 03_feature_engineering.ipynb
5. 04_baseline_model.ipynb
6. 05_model_comparison.ipynb
7. 06_final_analysis.ipynb

---

## Machine Learning Models Evaluated

The following regression algorithms were implemented and compared:

- Random Forest Regressor
- Extra Trees Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

---

## Performance Metrics

The models were evaluated using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (R²)

---

## Model Comparison

| Model | RMSE | MAE | R² |
|------|------:|------:|------:|
| Random Forest | 10.44 | 6.89 | 0.306 |
| Extra Trees | 10.56 | 6.99 | 0.289 |
| XGBoost | 10.83 | 7.15 | 0.252 |
| CatBoost | 11.14 | 7.28 | 0.209 |
| LightGBM | 11.18 | 7.28 | 0.203 |

---

## Best Performing Model

**Random Forest Regressor** achieved the best overall performance among all evaluated models.

Reasons:

- Lowest RMSE
- Lowest MAE
- Highest R² score

Feature importance analysis was performed to identify the molecular descriptors contributing most to prediction performance.

---

## Technologies Used

- Python
- Pandas
- NumPy
- RDKit
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Matplotlib
- Jupyter Notebook
- Git & GitHub

---

## Future Work

Potential improvements include:

- Hyperparameter optimization
- SHAP-based model interpretability
- Deep learning models
- Graph Neural Networks (GNNs)
- Molecular fingerprint-based models
- External dataset validation
- Web application deployment using Streamlit

---

## References

- NCI-ALMANAC Dataset
- RDKit Documentation
- Scikit-learn Documentation
- XGBoost Documentation
- LightGBM Documentation
- CatBoost Documentation

---

## Author

**Balaji Reddy**

M.Tech – Medical Biotechnology

Indian Institute of Technology Hyderabad

---

## Acknowledgements

This project was developed as part of **Discoverathon 2026**, focusing on machine learning approaches for drug combination synergy prediction.
