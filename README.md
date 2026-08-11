# Drug Combination Synergy Prediction using Machine Learning

## Project Overview

This project develops a machine learning pipeline for predicting **anticancer drug combination synergy scores** using the **NCI-ALMANAC** dataset.

The goal is to learn relationships between:

- Drug molecular properties
- Drug concentrations
- Cancer cell lines
- Cancer panels

and predict the synergy score of drug combinations.

The project combines:

- Large-scale biomedical data processing
- RDKit molecular descriptor generation
- Leakage-aware feature engineering
- Multiple machine learning regression models
- Cold-start evaluation
- Feature importance analysis
- SHAP-based model explainability
- Comprehensive regression, classification, and ranking evaluation
- Generalization-gap analysis
- Data-efficiency analysis

A major focus of the project is determining whether models can generalize beyond drug combinations, cell lines, and drugs observed during training.

---

# Problem Statement

Drug combination therapy is an important strategy in cancer treatment. Combining drugs can potentially improve therapeutic efficacy, overcome drug resistance, and enable lower doses of individual drugs.

However, experimentally testing large numbers of possible drug combinations is expensive and time-consuming.

Machine learning can help prioritize potentially effective drug combinations by learning patterns from previously measured experiments.

The objective of this project is to develop machine learning models that predict **drug combination synergy scores** and rigorously evaluate their performance under both conventional and realistic cold-start conditions.

---

# Objectives

The main objectives of this project are:

- Explore the NCI-ALMANAC drug combination dataset
- Clean and preprocess experimental data
- Generate molecular descriptors using RDKit
- Integrate molecular and biological features
- Detect and remove target leakage
- Create leakage-free train/validation/test splits
- Evaluate models under random and cold-start settings
- Compare multiple machine learning algorithms
- Evaluate regression, classification, and ranking performance
- Analyze model generalization
- Investigate the effect of training-data size
- Identify important predictive features
- Apply SHAP-based explainability
- Develop a reproducible machine learning workflow

---

# Dataset

## NCI-ALMANAC

The project uses the **NCI-ALMANAC** drug combination dataset from the National Cancer Institute.

### Raw Dataset

The raw dataset contains:

- **3,686,475 experimental rows**
- **105 unique drugs**
- **5,460 initial drug pairs**
- **61 initial cell lines**

After removing rows with missing synergy scores:

- **2,871,444 valid rows**
- **5,242 unique drug pairs**
- **105 unique drugs**
- **60 cancer cell lines**

### Target Variable

The prediction target is:

```text
SCORE
