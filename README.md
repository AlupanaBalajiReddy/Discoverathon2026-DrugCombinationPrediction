# Drug Combination Synergy Prediction

## Project Overview

This project predicts drug combination synergy scores using machine learning.

Drug combinations are widely used in cancer therapy to improve treatment effectiveness while reducing toxicity.

The project uses the NCI-ALMANAC drug combination dataset and molecular descriptors generated using RDKit.

Multiple regression models are developed and compared to predict synergy scores.


## Objectives

- Explore the NCI-ALMANAC dataset
- Clean and preprocess data
- Generate molecular descriptors using RDKit
- Train baseline machine learning models
- Compare different regression algorithms
- Identify important molecular features

## Dataset

NCI-ALMANAC

## Project Workflow

Raw Dataset

↓

Data Cleaning

↓

Feature Engineering

↓

RDKit Molecular Descriptors

↓

Baseline Random Forest

↓

Model Comparison

↓

Final Analysis

## Folder Structure

Discoverathon2026/
│
├── data/
│   ├── raw/
│   ├── external/
│   └── processed/
│
├── notebooks/
│
├── models/
│
├── results/
│
├── src/
│
├── README.md
├── PROJECT_LOG.md
└── requirements.txt


## Technologies Used

Python

Pandas

NumPy

RDKit

Scikit-Learn

Jupyter Notebook

Git

GitHub

## Baseline Model Results

Random Forest

RMSE : 9.48

MAE : 6.30

R² : 0.42

## Future Work

Train XGBoost

Train LightGBM

Train CatBoost

Hyperparameter Optimization

Model Explainability (SHAP)

Deployment

## How to Run

git clone ...

pip install -r requirements.txt

Run notebooks in order

00

↓

01

↓

02

↓

03

↓

04

↓

05

↓

06

## Author

Balaji Reddy

M.Tech Medical Biotechnology

IIT Hyderabad