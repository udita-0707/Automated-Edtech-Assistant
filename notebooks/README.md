# Data Analysis & Exploratory Notebooks

This directory contains standalone analytical scripts used to characterize the `SciEntsBank` dataset and validate feature engineering decisions.

## Contents
- **`eda.py`**: The primary executable analysis script. It generates all core visualizations and statistical insights used in the project report.
- **Generated Visuals**:
  - `label_distribution.png`: 3-way distribution showing dataset imbalance.
  - `text_length_distribution.png`: Histogram proof of answer variability.
  - `correlation_matrix.png`: Verifying the orthogonality of engineered features.
  - `pca_visual_proof.png`: 2D projection confirming semantic manifold separability.

## Role in Pipeline
The logic here is decoupled from the production `ml-service` to maintain a lightweight microservice. These scripts are intended for research, feature validation, and academic documentation.
