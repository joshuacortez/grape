import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

sys.path.append("..")
from regression_model import RegressionModel
from hyperparameter_optimizer import HyperParameterOptimizer

def main():

    # get sample data from ATM analysis
    df = pd.read_csv("../../data/features2.csv", index_col = 0)
    df["log_txn"] = df["Transactions"].apply(np.log)

    df_train, df_test = train_test_split(
        df, 
        test_size = 0.2,
        random_state = 2019,
    )

    feat_cols = list(df.columns)
    feat_cols.remove("Transactions")
    feat_cols.remove("log_txn")

    # model = RegressionModel("random_forest")
    # model = RegressionModel("elastic_net")
    # model = RegressionModel("lightgbm")
    model = RegressionModel("xgboost")
    hpo = HyperParameterOptimizer(
        verbosity = 2,
        override_params = {
            "n_estimators": 100
        },
        seed = 2019,
    )

    optimized_model = hpo.fit_optimize(
        model,
        df_train.loc[:, feat_cols].astype('float64'),
        df_train.loc[:, "log_txn"].astype('float64'),
        max_evals = 10,
    )

    print(type(optimized_model))
    print(optimized_model.feature_importance_df)

if __name__ == "__main__":
    main()
    main()