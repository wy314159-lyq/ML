# MatSci-ML Studio

A Python-Based Machine Learning GUI for Materials Science.

## Overview

MatSci-ML Studio is a comprehensive, user-friendly, Python-based desktop application with a Graphical User Interface (GUI) specifically designed for machine learning tasks in the field of materials science. The application guides the user through the entire ML workflow, from data ingestion and preprocessing to model training, hyperparameter optimization, feature selection, evaluation, and prediction. A core principle is the rigorous use of `sklearn.pipeline.Pipeline` to prevent data leakage and ensure reproducible, scientifically sound workflows.

## Target Audience

Materials scientists and researchers who may not be expert programmers but need a robust tool to apply machine learning to their data.

## Modules

1.  **Data Ingestion & Preprocessing:** Import, explore, clean, preprocess, and visualize data. Define features (X) and target (y).
2.  **Feature Engineering, Model Selection & Advanced Feature Selection:** Select task type, benchmark models, configure training, and apply multi-stage feature selection.
3.  **Final Model Training, Hyperparameter Optimization & Evaluation:** Perform HPO on selected models, train final pipelines, and evaluate comprehensively.
4.  **Prediction & Results Export:** Load trained models, predict on new data, and export results.

## Technical Requirements

*   **Programming Language:** Python 3.x
*   **GUI Framework:** PySide2 (or PyQt5)
*   **Core Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn.
*   **Optional Libraries:** XGBoost, LightGBM, Optuna/Hyperopt, SHAP, LIME, mlxtend.
*   **Strict Pipeline Usage:** All preprocessing and modeling within `sklearn.pipeline.Pipeline`.
*   **English-Only Output:** All code, comments, and GUI text in English.

## Getting Started

(Instructions to be added)

## Contributing

(Guidelines for contributing to be added)

## License

(License information to be added) 