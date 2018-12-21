from models.regression_model import RegressionModel
from models.hyperparameter_optimizer import HyperparameterOptimizer
from models.params_space import model_params_space_dict
from preprocessing import Preprocessor

import csv

import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer

import hyperopt

import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import datetime
from timeit import default_timer as timer

def test_regression_model():
		from sklearn.model_selection import train_test_split
		from sklearn.datasets import load_boston
		from sklearn.metrics import r2_score
		RANDOM_SEED =  46257
		
		boston = load_boston()
		df = pd.DataFrame(data = boston["data"])
		df.columns = ["var_{}".format(col) for col in df.columns]
		df["target"] = boston["target"]
		train_df, test_df = train_test_split(df, test_size = 0.2, random_state = RANDOM_SEED)
		
		attribute_cols_list = [col for col in df.columns if col != "target"]
		y_cols_list = ["target"]
		USE_WEIGHT_SAMPLES = False
		categorical_vars_list = []
		
		for model_type in ["elastic_net", "random_forest", "xgboost", "lightgbm"]:
				print("fitting baseline {} model".format(model_type))
				model = RegressionModel(model_type = model_type, random_state = RANDOM_SEED,
											metric = "l2")
				model.fit(X_train = train_df.loc[:,attribute_cols_list],
								 y_train = train_df["target"])
		
				predictions = model.predict(X_test = test_df.loc[:,attribute_cols_list])
				r2 = r2_score(test_df["target"], predictions)
				print("R-Squared on test set predictions on boston dataset is {}".format(r2))

def test_hyperparameter_optimizer():
		from sklearn.model_selection import train_test_split
		from sklearn.datasets import load_boston
		RANDOM_SEED =  46257

		boston = load_boston()
		df = pd.DataFrame(data = boston["data"])
		df.columns = ["var_{}".format(col) for col in df.columns]
		df["target"] = boston["target"]
		train_df, test_df = train_test_split(df, test_size = 0.2, random_state = RANDOM_SEED)
		train_valid_folds = KFold(n_splits = 5, shuffle = True, random_state = RANDOM_SEED)

		attribute_cols_list = [col for col in df.columns if col != "target"]
		y_cols_list = ["target"]
		USE_WEIGHT_SAMPLES = False
		categorical_vars_list = []

		for model_type in ["elastic_net", "random_forest", "lightgbm", "xgboost"]:
				hpo = HyperparameterOptimizer(model_type = model_type, 
																			X_train = train_df.loc[:,attribute_cols_list],
																		 y_train = train_df["target"],
																		 y_colname = "target",
																		 metric = "l2",
																		 random_state = RANDOM_SEED,
																			train_valid_folds = train_valid_folds,
																			output_folder = None
																		 )
				print(model_type)
				results = hpo.run_hyperparameter_tuning(parameter_space_dict = model_params_space_dict[model_type], max_evals = 10)
				results = pd.DataFrame(results)
				print(results.head())

def test_preprocessing():
		from sklearn.model_selection import train_test_split
		from sklearn.datasets import load_boston
		RANDOM_SEED =  46257
		
		boston = load_boston()
		df = pd.DataFrame(data = boston["data"])
		df.columns = ["var_{}".format(col) for col in df.columns]
		df["target"] = boston["target"]
		train_df, test_df = train_test_split(df, test_size = 0.2, random_state = RANDOM_SEED)
		
		
		attribute_cols_list = [col for col in df.columns if col != "target"]
		y_cols_list = ["target"]
		USE_WEIGHT_SAMPLES = False
		categorical_vars_list = []
		
		preprocessor = Preprocessor(model_type = "elastic_net")
		preprocessed_df = preprocessor.fit_transform(train_df.loc[:,attribute_cols_list], categorical_cols_list=[])
		print(preprocessed_df.head())


if __name__ == "__main__":
	test_regression_model()
	test_hyperparameter_optimizer()
	test_preprocessing()