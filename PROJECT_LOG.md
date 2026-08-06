# Discoverathon 2026

## Problem Statement

Challenge 1: Predicting Drug Combination Activity

## Team

### Technical Lead
Your Name

Responsibilities
- Machine Learning
- Feature Engineering
- Model Development
- Integration

### Research Mentor
Mentor Name

Responsibilities
- Scientific Guidance
- Biological Interpretation
- Report Review

### Dataset Engineer
Student 1

Responsibilities
- Dataset Collection
- Data Cleaning
- Exploratory Data Analysis

### Documentation Engineer
Student 2

Responsibilities
- Literature Review
- Report Writing
- Presentation
- References

---

## Day 1

Created project structure.

Initialized Git repository.

Created GitHub repository.



# Project Log

---

## August 05, 2026

### Completed

#### Dataset Exploration
- Performed exploratory data analysis on the NCI-ALMANAC dataset.
- Identified missing values and duplicate records.
- Analyzed the distribution of the synergy score (SCORE).

#### Data Preprocessing
- Removed single-drug experiments.
- Removed records with missing SCORE values.
- Selected relevant columns for modeling.
- Extracted unique drug identifiers.

#### Feature Engineering
- Downloaded the official NCI SMILES dataset.
- Merged NSC identifiers with SMILES.
- Removed compounds with missing or invalid SMILES.
- Generated molecular descriptors using RDKit.
- Created descriptors for Drug 1 and Drug 2.
- Merged descriptors with the experimental dataset.
- Generated the final feature-engineered dataset.

#### Baseline Machine Learning Model
- Built a Random Forest Regressor.
- Performed an 80/20 train-test split.
- Evaluated model performance.

##### Results

| Metric | Value |
|--------|-------|
| RMSE | 9.48 |
| MAE | 6.30 |
| R² | 0.42 |

#### Version Control
- Organized the repository.
- Added Git ignore rules for large files.
- Committed changes.
- Pushed the project to the `develop` branch.

---

## Next Tasks

- Train Extra Trees Regressor.
- Train XGBoost Regressor.
- Train LightGBM Regressor.
- Compare all models.
- Select the best-performing model.
- Perform final analysis.