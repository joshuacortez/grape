# GRAPE
GRAPE is a regression API in Python environment

# Description
GRAPE makes it easy to fit a regression model with hyperparameter optimization. It strings together the workflow of model fitting, hyperparameter tuning, and model diagnostics. (model interpretability coming soon!).
- Available Regression Methods
1. Elastic Net (from sklearn)
2. Random Forest (from sklearn)
3. xgboost
4. lightgbm
- Hyperparameter Optimization
    - Grape Uses Hyperopt's tree parzen estimator

# Installation
- `pip install grape-model`
- If you're having trouble installing some of the dependencies (especially lightgbm, xgboost), try installing them via `conda-forge` before installing GRAPE 

# Sample Usage
See `sample_grape_use_case.ipynb`