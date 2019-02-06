import csv

import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV

import hyperopt

import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import datetime
from timeit import default_timer as timer
from copy import copy

from .param_space import parameter_space

class HyperParameterOptimizer:

    def __init__(self, **kwargs):
        self._random_state = kwargs.get("random_state", None)

    def _objective_random_forest(self, model_params):
        '''
        model_params dictionary is provided by hyperopt.fmin()
        '''
        start = timer()

        model_params_ = copy(model_params)

        model_params_["criterion"] = self._model._metric
        for param in ["min_samples_split", "min_samples_leaf", "n_estimators"]:
            if param in model_params_.keys():
                model_params_[param] = int(model_params_[param])

        model = RandomForestRegressor(n_jobs = -1, random_state = self._random_state, **model_params)

        oob_score = model_params_.get("oob_score", False)
        if oob_score:
            model.fit(
                X = self._X_train, 
                y = self._y_train, 
                sample_weight = self._sample_weight
            )
            if self._model._metric == "mae":
                loss = mean_absolute_error(y_true = self._y_train, y_pred = model.oob_prediction_)
            if self._model._metric == "mse":
                loss = mean_squared_error(y_true = self._y_train, y_pred = model.oob_prediction_)
        else:
            cross_val_scores = cross_val_score(
                estimator = model,
                X = self._X_train,
                y = self._y_train,
                scoring = self._scoring,
                cv = self._train_valid_folds_x,
                fit_params = {"sample_weight":self._sample_weight},
                n_jobs = -1
            )
            loss = np.mean(cross_val_scores)*(-1)
        
        self.num_iterations += 1
        run_time = timer() - start 
        return {
            'loss': loss, 
            'params': model_params_, 
            "num_iterations": self.num_iterations, 
            "train_time": run_time,
            "status": hyperopt.STATUS_OK,
        }

    def fit_optimize(self, model_, X_train, y_train, **kwargs):
        self._sample_weight = kwargs.get("sample_weight", None)
        train_valid_folds = kwargs.get("train_valid_folds", None)

        bayes_trials = hyperopt.Trials()

        self._model = model_
        self._X_train = X_train
        self._y_train = y_train
        if train_valid_folds is not None:
            self._train_valid_folds_x = list(train_valid_folds.split(X_train))

        if model_._model_type == "random_forest":
            objective = self._objective_random_forest

            if self._model._metric == "mae":
                self._scoring = "neg_mean_absolute_error"
            elif self._model._metric == "mse":
                self._scoring = "neg_mean_squared_error"

        self.num_iterations = 0
