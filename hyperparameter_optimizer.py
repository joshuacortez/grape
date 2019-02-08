import csv

import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV

import hyperopt

import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import datetime
from timeit import default_timer as timer
from copy import copy

from param_space import parameter_space
from utils import get_elastic_net_l1_ratio

class HyperParameterOptimizer:

    def __init__(self, **kwargs):
        self._random_state = kwargs.get("random_state", None)
        self.verbosity = kwargs.get("verbosity", 0)

    def _print_iter(self):
        if self.verbosity >= 2:
            print("Iteration:", self.num_iterations)

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

        model = RandomForestRegressor(n_jobs = -1, random_state = self._random_state, **model_params_)

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

        self._print_iter()

        run_time = timer() - start 
        return {
            'loss': loss, 
            'params': model_params_, 
            "num_iterations": self.num_iterations, 
            "train_time": run_time,
            "status": hyperopt.STATUS_OK,
        }

    def _objective_elastic_net(self, model_params):
        start = timer()

        model_params_ = copy(model_params)
        model_params_["l1_ratio"] = get_elastic_net_l1_ratio(model_params)

        model = ElasticNet(
            max_iter = 10000,
            random_state = self._random_state,
            **model_params_
        )

        pipeline = make_pipeline(StandardScaler(), model)
        cross_val_scores = cross_val_score(
            estimator = pipeline,
            X = self._X_train,
            y = self._y_train,
            scoring = self._scoring,
            cv = self._train_valid_folds_x,
            n_jobs = -1
        )
        loss = np.mean(cross_val_scores)*(-1)

        self.num_iterations += 1

        self._print_iter()

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
        max_evals = kwargs.get("max_evals", 100)

        train_valid_folds = kwargs.get("train_valid_folds", None)

        bayes_trials = hyperopt.Trials()

        self._model = model_
        self._X_train = X_train
        self._y_train = y_train
        if train_valid_folds is not None:
            self._train_valid_folds_x = list(train_valid_folds.split(X_train))
        else:
            self._train_valid_folds_x = None

        if model_._model_type == "random_forest":
            objective = self._objective_random_forest

            if self._model._metric == "mae":
                self._scoring = "neg_mean_absolute_error"
            elif self._model._metric == "mse":
                self._scoring = "neg_mean_squared_error"
        elif model_._model_type == "elastic_net":
            # TODO: just use ElasticNetCV instead of hyperopt.fmin()

            objective = self._objective_elastic_net
            
            if self._model._metric == "mae":
                self._scoring = "neg_mean_absolute_error"
            elif self._model._metric == "mse":
                self._scoring = "neg_mean_squared_error"
        
        # TODO: implement other model types

        self.num_iterations = 0

        self.best_params = hyperopt.fmin(
            fn = objective,
            space = parameter_space[self._model._model_type],
            algo = hyperopt.tpe.suggest,
            max_evals = max_evals,
            trials = bayes_trials,
            rstate = np.random.RandomState(self._random_state),
        )

        # return an optimized model
        if model_._model_type == "random_forest":
            for param in ["min_samples_split", "min_samples_leaf", "n_estimators"]:
                if param in self.best_params.keys():
                    self.best_params[param] = int(self.best_params[param])

            best_model = RandomForestRegressor(
                n_jobs = -1, 
                random_state = self._random_state, 
                **self.best_params
            )
            best_model.fit(
                X = self._X_train, 
                y = self._y_train, 
                sample_weight = self._sample_weight
            )
        elif model_._model_type == "elastic_net":

            self.best_params.pop("penalty_type")

            best_model = ElasticNet(
                max_iter = 10000,
                random_state = self._random_state,
                **self.best_params
            )
            best_model.fit(
                X = self._X_train, 
                y = self._y_train, 
            )

        return best_model
        
        # TODO: implement other model types


