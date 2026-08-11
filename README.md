# Drug Combination Synergy Prediction using Machine Learning

## Project Overview

This project develops a machine learning pipeline for predicting **anticancer drug combination synergy scores** using the **NCI-ALMANAC** dataset.

The goal is to learn relationships between drug molecular properties, drug concentrations, cancer cell lines, and cancer panels in order to predict the synergy score of drug combinations.

The project combines:

- Large-scale biomedical data processing
- RDKit molecular descriptor generation
- Leakage-aware feature engineering
- Multiple machine learning regression models
- Cold-start evaluation
- Feature importance analysis
- SHAP-based model explainability

A major focus of the project is evaluating whether a model can generalize beyond combinations, cell lines, and drugs that were observed during training.

---

# Problem Statement

Drug combination therapy is an important strategy in cancer treatment. Combining drugs can potentially improve therapeutic efficacy, overcome drug resistance, and enable lower doses of individual drugs.

However, experimentally testing large numbers of possible drug combinations is expensive and time-consuming.

Machine learning can be used to prioritize potentially effective combinations by learning patterns from previously measured drug combination experiments.

The objective of this project is therefore to develop machine learning models that predict **drug combination synergy scores** and evaluate how well these models generalize to previously unseen biological conditions.

---

# Objectives

The main objectives of this project are:

- Explore the NCI-ALMANAC drug combination dataset
- Clean and preprocess the experimental data
- Generate molecular descriptors using RDKit
- Integrate molecular and biological features
- Detect and remove target leakage
- Create realistic train/validation/test splits
- Evaluate models under random and cold-start settings
- Compare multiple regression algorithms
- Identify important predictive features
- Apply SHAP-based explainability
- Develop a reproducible machine learning workflow

---

# Dataset

## NCI-ALMANAC

The project uses the **NCI-ALMANAC** drug combination dataset from the National Cancer Institute.

The raw dataset contains:

- **3,686,475 experimental rows**
- **105 unique drugs**
- **5,460 initial drug pairs**
- **61 initial cell lines**

After removing rows with missing synergy scores:

- **2,871,444 valid rows**
- **5,242 unique drug pairs**
- **60 cancer cell lines**
- **105 unique drugs**

### Target Variable

The prediction target is:

```text
SCORE
