# Analytics Module

## Overview

This module uses the Titanic dataset to perform exploratory data analysis and basic machine learning.

The analysis covers:
- Data cleaning
- Missing value handling
- Univariate analysis
- Bivariate analysis
- Correlation analysis
- Multivariate analysis
- Feature standardization
- Classification models

## Files

### `01_eda.py`

Performs the main exploratory data analysis.

It includes:
- Missing value analysis
- Age and fare distributions
- Outlier detection using IQR
- Survival analysis by gender and passenger class
- Correlation heatmap
- Family size analysis
- Standardization of age and fare

### `02_modeling.py`

Builds classification models to predict passenger survival.

The models used are:

- Logistic Regression
- Random Forest

The models are compared using:
- Accuracy
- Precision
- Recall
- F1 Score

A confusion matrix is also generated for each model.

## How to Run

Install the required packages:

```bash
pip install -r ../requirements.txt
