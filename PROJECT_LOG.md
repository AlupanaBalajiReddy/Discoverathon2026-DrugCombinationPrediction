# PROJECT LOG
# Drug Combination Synergy Prediction using Machine Learning

## Project Title

**Drug Combination Synergy Prediction using Machine Learning**

---

# 1. Project Goal

The goal of this project is to develop a machine learning pipeline capable of predicting anticancer drug combination synergy scores using molecular descriptors and biological information from the NCI-ALMANAC dataset.

The project also evaluates whether the developed models can generalize to previously unseen:

- Drug combinations
- Cancer cell lines
- Drugs

---

# 2. Dataset Exploration

The NCI-ALMANAC dataset was inspected to understand its structure, variables, missing values, and target distribution.

Initial dataset statistics:

- Raw rows: 3,686,475
- Raw columns: 29
- Unique drugs: 105
- Initial unique drug pairs: 5,460
- Initial unique cell lines: 61

The target variable was identified as:

```text
SCORE
