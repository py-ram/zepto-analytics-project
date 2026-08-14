# Analytics Module

## Overview

This module uses the Titanic dataset for exploratory data analysis and basic machine learning.

The analysis covers:

- Data cleaning
- Missing value handling
- Univariate analysis
- Bivariate analysis
- Correlation analysis
- Multivariate analysis
- Outlier detection
- Feature standardization
- Classification models

## Files

### `01_eda.py`

Performs exploratory data analysis on the Titanic dataset.

It includes:

- Missing value analysis
- Age and fare distributions
- IQR-based outlier detection
- Survival analysis by gender and passenger class
- Correlation analysis
- Family size analysis
- Standardization of age and fare

### `02_modeling.py`

Builds classification models to predict passenger survival.

The models used are:

- Logistic Regression
- Decision Tree
- Random Forest

The models are compared using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

Confusion matrices and ROC curves are also generated.

## How to Run

From this folder:

```bash
python 01_eda.py
python 02_modeling.py