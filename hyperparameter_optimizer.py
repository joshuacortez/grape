import csv

import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV

import hyperopt

import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import datetime
from timeit import default_timer as timer
from copy import copy

from regression_model import RegressionModel
from param_space import parameter_space
from utils import get_elastic_net_l1_ratio, _huber_approx_obj

class HyperParameterOptimizer:


    def __init__(self, **kwargs):
        self._seed = kwargs.get("seed", None)
        self.verbosity = kwargs.get("verbosity", 0)
        self.diagnose_hyperparam_optim = kwargs.get("diagnose_hyperparam_optim",False)
        self.hyperparam_optim_summary = []

        self._random_state = np.random.RandomState(self._seed)

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

        model = RandomForestRegressor(
            n_jobs = -1, 
            random_state = self._random_state, 
            **model_params_
        )

        oob_score = model_params_.get("oob_score", False)
        if oob_score:
            model.fit(
                X = self._X_train, 
                y = self._y_train, 
                sample_weight = self._sample_weight
            )
            if self._model._metric == "mae":
                loss = mean_absolute_error(
                    y_true = self._y_train, 
                    y_pred = model.oob_prediction_
                )
            if self._model._metric == "mse":
                loss = mean_squared_error(
                    y_true = self._y_train, 
                    y_pred = model.oob_prediction_
                )
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

        output =  {
                'loss': loss, 
                'params': model_params_, 
                "num_iterations": self.num_iterations, 
                "train_time": run_time
            }

        if self.diagnose_hyperparam_optim:
            self.hyperparam_optim_summary.append(output.copy())

        output["status"] = hyperopt.STATUS_OK

        return output

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

        output = {
            'loss': loss, 
            'params': model_params_, 
            "num_iterations": self.num_iterations, 
            "train_time": run_time,
        }

        if self.diagnose_hyperparam_optim:
            self.hyperparam_optim_summary.append(output.copy())

        output["status"] = hyperopt.STATUS_OK

        return output

    def _objective_lightgbm(self, model_params):
        start = timer()

        model_params_ = copy(model_params)
        model_params_["objective"] = self._model._metric

        # conditional sampling from bayesian domain for the goss bossting type
        if "boosting_type" in model_params_.keys():
            subsample = model_params_['boosting_type'].get('subsample', 1.0)

            model_params_['boosting_type'] = model_params_['boosting_type']['boosting_type']
            model_params_['subsample'] = subsample

        for param in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
            if param in model_params_.keys():
                model_params_[param] = int(model_params_[param])

        cv_results = lgb.cv(
            params = model_params_, 
            train_set = self._d_train,
            num_boost_round = 10000, 
            folds = self._train_valid_folds_x, 
            # nfold = 10,
            early_stopping_rounds = 100, 
            seed = self._random_state, 
            fobj = None, 
            feval = None
        )

        metric_mean_name = [key for key in cv_results.keys() if "-mean" in key][0]
        metric_std_name = [key for key in cv_results.keys() if "-std" in key][0]

        loss = np.min(cv_results[metric_mean_name])
        std_loss = cv_results[metric_std_name][np.argmin(cv_results[metric_mean_name])]
        num_boost_round = int(np.argmin(cv_results[metric_mean_name]) + 1)

        self.num_iterations += 1

        self._print_iter()

        run_time = timer() - start 

        output = {
            'loss': loss, 
            "std_loss":std_loss,
            'params': model_params_, 
            "num_iterations": self.num_iterations,
            'estimators': num_boost_round,
            "train_time": run_time,
        }

        if self.diagnose_hyperparam_optim:
            self.hyperparam_optim_summary.append(output.copy())

        output["status"] = hyperopt.STATUS_OK

        return output

    def _objective_xgboost(self, model_params):
        start = timer()

        model_params_ = copy(model_params)

        if self._model._metric == "mae":
            obj = _huber_approx_obj
            metric = "mae"

        elif self._model._metric == "mse":
            model_params_["objective"] = "reg:linear"
            obj = None
            metric = "rmse"

        for parameter_name in ['min_child_weight']:
            if parameter_name in model_params_.keys():
                model_params_[parameter_name] = int(model_params_[parameter_name])

        model_params_["silent"] = 1

        cv_results = xgb.cv(
            params = model_params_, 
            dtrain = self._d_train, 
            num_boost_round = 100000, 
            folds = self._train_valid_folds_x, 
            early_stopping_rounds = 100, 
            seed = self._seed, 
            obj = obj, 
            metrics = metric
        )

        metric_mean_name = "test-{}-mean".format(metric)
        metric_std_name = "test-{}-std".format(metric)

        best_idx = cv_results[metric_mean_name].idxmin()
        loss = cv_results.loc[best_idx,metric_mean_name]
        std_loss = cv_results.loc[best_idx,metric_std_name]

        num_boost_round = best_idx + 1

        self.num_iterations += 1
        self._print_iter()

        run_time = timer() - start

        output = {
            'loss': loss, 
            "std_loss":std_loss,
            'params': model_params_, 
            "num_iterations": self.num_iterations,
            'num_boost_round': num_boost_round,
            "train_time": run_time,
        }

        if self.diagnose_hyperparam_optim:
            self.hyperparam_optim_summary.append(output.copy())

        output["status"] = hyperopt.STATUS_OK

        return output

    def fit_optimize(self, model_, X_train, y_train, **kwargs):
        self._sample_weight = kwargs.get("sample_weight", None)
        self._override_params = kwargs.get("override_params", {})

        max_evals = kwargs.get("max_evals", 100)

        train_valid_folds = kwargs.get("train_valid_folds", None)

        bayes_trials = hyperopt.Trials()

        self._model = model_
        self._X_train = X_train
        self._y_train = y_train
        if train_valid_folds is not None:
            self._train_valid_folds_x = list(train_valid_folds.split(X_train))
        elif model_._model_type in ["lightgbm", "lightgbm_special"]:
            # NOTE: default splitter of lightgbm doesn't work
                # providing my own default instead
            default_splitter = KFold(
                n_splits = 5, 
                shuffle = True, 
                random_state = self._random_state
            )
            self._train_valid_folds_x = list(default_splitter.split(X_train))
        else:
            default_splitter = KFold(
                n_splits = 5, 
                shuffle = True, 
                random_state = self._random_state
            )
            self._train_valid_folds_x = list(default_splitter.split(X_train))

        model_params = copy(parameter_space[self._model._model_type])

        if model_._model_type == "random_forest":
            objective = self._objective_random_forest

            if self._model._metric == "mae":
                self._scoring = "neg_mean_absolute_error"
            elif self._model._metric == "mse":
                self._scoring = "neg_mean_squared_error"

            # model_params["random_state"] = self._random_state

        elif model_._model_type == "elastic_net":
            # TODO: just use ElasticNetCV instead of hyperopt.fmin()

            objective = self._objective_elastic_net
            
            if self._model._metric == "mae":
                self._scoring = "neg_mean_absolute_error"
            elif self._model._metric == "mse":
                self._scoring = "neg_mean_squared_error"
        elif model_._model_type in ["lightgbm", "lightgbm_special"]:
            self._d_train = lgb.Dataset(
                data = self._X_train, 
                label = self._y_train, 
                weight = self._sample_weight
            )
            
            # TODO: make lightgbm silent pls
            objective = self._objective_lightgbm
        elif model_._model_type == "xgboost":
            self._d_train = d_train = xgb.DMatrix(
                data = self._X_train, 
                label = self._y_train, 
                weight = self._sample_weight
            )

            objective = self._objective_xgboost

            # model_params["seed"] = self._seed

        self.num_iterations = 0

        self.best_params = hyperopt.fmin(
            fn = objective,
            space = model_params,
            algo = hyperopt.tpe.suggest,
            max_evals = max_evals,
            trials = bayes_trials,
            rstate = self._random_state,
        )

        # return an optimized model
        # TODO: return grape RegressionModel object
            # instead of some other class
        if model_._model_type == "random_forest":
            for param in ["min_samples_split", "min_samples_leaf", "n_estimators"]:
                if param in self.best_params.keys():
                    self.best_params[param] = int(self.best_params[param])

            model_params = copy(self.best_params)
            model_params.update(self._override_params)

            best_model = RandomForestRegressor(
                n_jobs = -1, 
                random_state = self._random_state, 
                **model_params
            )
            best_model.fit(
                X = self._X_train, 
                y = self._y_train, 
                sample_weight = self._sample_weight
            )
        elif model_._model_type == "elastic_net":

            self.best_params.pop("penalty_type")

            model_params = copy(self.best_params)
            model_params.update(self._override_params)

            best_model = ElasticNet(
                max_iter = 10000,
                random_state = self._random_state,
                **model_params
            )
            best_model.fit(
                X = self._X_train, 
                y = self._y_train, 
            )
        
        elif model_._model_type in ["lightgbm", "lightgbm_special"]:
            for param in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
                if param in self.best_params.keys():
                    self.best_params[param] = int(self.best_params[param])
            
            model_params = copy(self.best_params)
            model_params.update(self._override_params)

            best_model = lgb.train(
                params = model_params,
                train_set = self._d_train, 
                fobj = None,
                feval = None
            )

        elif model_._model_type == "xgboost":
            if self._model._metric == "mae":
                obj = _huber_approx_obj
                metric = "mae"
            elif self._model._metric == "mse":
                self.best_params["objective"] = "reg:linear"
                obj = None
                metric = "rmse"
                
            for parameter_name in ['min_child_weight']:
                if parameter_name in self.best_params.keys():
                    self.best_params[parameter_name] = int(self.best_params[parameter_name])

            self.best_params["silent"] = 1
            
            model_params = copy(self.best_params)
            model_params.update(self._override_params)

            best_model = xgb.train(
                params = model_params, 
                dtrain = self._d_train, 
                obj = obj, 
                num_boost_round = parameter_space.get("num_boost_round", 200)
            )

        # NOTE: return a regression model object
        # not the class from whatever package
        reg_model = RegressionModel.from_trained(
            best_model,
            feature_names = X_train.columns,
        )
        return reg_model
        


