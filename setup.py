import setuptools

long_description = """
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
"""

setuptools.setup(
    name="grape-model", 
    version="0.0.3",
    author="Joshua Cortez",
    author_email="joshua.m.cortez@gmail.com",
    description="GRAPE makes it easy to fit a regression model with hyperparameter optimization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joshuacortez/grape",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[
        "numpy",
        "scikit-learn",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "xgboost",
        "lightgbm",
        "hyperopt"
    ],
    python_requires='>=3.6',
)