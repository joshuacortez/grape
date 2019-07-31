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
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime
from timeit import default_timer as timer
from copy import copy

from regression_model import RegressionModel
from param_space import parameter_space
from utils import get_elastic_net_l1_ratio, _huber_approx_obj, eval_func_is_higher_better

# todo: check what loss_variance actually does

class HyperParameterOptimizer:

    def __init__(self, eval_func_name = "mse", random_seed = 93375481, verbosity = 0):
        self._random_seed = random_seed
        self.verbosity = verbosity

        assert isinstance(eval_func_name,str), "eval_func_name should be a string"
        self.eval_func_name = eval_func_name
        self.eval_func_is_higher_better = eval_func_is_higher_better[self.eval_func_name]

        self.random_seed = random_seed

    def _print_iter(self):
        if self.verbosity >= 2:
            print("Iteration:", self.num_iterations)

    def _objective_elastic_net(self, model_params):
        start = timer()

        model_params_ = copy(model_params)
        model_params_["l1_ratio"] = get_elastic_net_l1_ratio(model_params_)

        cv_scores = self.model.cross_validate(self._train_valid_folds,
                                              self.eval_func_name,
                                                model_params_)

        loss = cv_scores["cv-{}-mean".format(self.eval_func_name)]
        loss_std = cv_scores["cv-{}-std".format(self.eval_func_name)]
        if self.eval_func_is_higher_better:
            loss = loss*(-1)

        self.num_iterations += 1

        self._print_iter()

        run_time = timer() - start 

        output = {
            'loss': loss, 
            'loss_variance':loss_std**2,
            'params': model_params_, 
            "num_iterations": self.num_iterations, 
            "iteration_run_time": run_time,
        }

        self.hyperopt_summary.append(output.copy())

        output["status"] = hyperopt.STATUS_OK

        return output

    def _objective_random_forest(self, model_params):
        '''
        model_params dictionary is provided by hyperopt.fmin()
        '''
        start = timer()

        model_params_ = copy(model_params)

        for param in ["min_samples_split", "min_samples_leaf", "n_estimators"]:
            if param in model_params_.keys():
                model_params_[param] = int(model_params_[param])

        cv_scores = self.model.cross_validate(self._train_valid_folds,
                                              self.eval_func_name,
                                                model_params_)

        loss = cv_scores["cv-{}-mean".format(self.eval_func_name)]
        loss_std = cv_scores["cv-{}-std".format(self.eval_func_name)]
        if self.eval_func_is_higher_better:
            loss = loss*(-1)
        
        self.num_iterations += 1

        self._print_iter()

        run_time = timer() - start 

        output =  {
                'loss': loss, 
                'loss_variance': loss_std**2,
                'params': model_params_, 
                "num_iterations": self.num_iterations, 
                "iteration_run_time": run_time
            }

        self.hyperopt_summary.append(output.copy())

        output["status"] = hyperopt.STATUS_OK

        return output

    def _objective_lightgbm(self, model_params):
        start = timer()

        model_params_ = copy(model_params)

        # conditional sampling from bayesian domain for the goss bossting type
        if "boosting_type" in model_params_.keys():
            subsample = model_params_['boosting_type'].get('subsample', 1.0)

            model_params_['boosting_type'] = model_params_['boosting_type']['boosting_type']
            model_params_['subsample'] = subsample

        for param in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
            if param in model_params_.keys():
                model_params_[param] = int(model_params_[param])

        cv_scores = self.model.cross_validate(self._train_valid_folds,
                                        self.eval_func_name,
                                        model_params_)

        loss = cv_scores["cv-{}-mean".format(self.eval_func_name)]
        loss_std = cv_scores["cv-{}-std".format(self.eval_func_name)]

        if self.eval_func_is_higher_better:
            loss = np.array(loss)*(-1)

        best_idx_loss = np.argmin(loss)
        loss = loss[best_idx_loss]
        loss_std = loss_std[best_idx_loss]
        num_boost_round = best_idx_loss + 1
        

        self.num_iterations += 1

        self._print_iter()

        run_time = timer() - start 

        output = {
            'loss': loss, 
            "loss_variance":loss_std**2,
            'params': model_params_, 
            "num_iterations": self.num_iterations,
            'estimators': num_boost_round,
            "iteration_run_time": run_time,
        }

        self.hyperopt_summary.append(output.copy())

        output["status"] = hyperopt.STATUS_OK

        return output

    def _objective_xgboost(self, model_params):
        start = timer()

        model_params_ = copy(model_params)

        for parameter_name in ['min_child_weight']:
            if parameter_name in model_params_.keys():
                model_params_[parameter_name] = int(model_params_[parameter_name])

        cv_scores = self.model.cross_validate(self._train_valid_folds,
                                        self.eval_func_name,
                                        model_params_)
        loss = cv_scores["cv-{}-mean".format(self.eval_func_name)]
        loss_std = cv_scores["cv-{}-std".format(self.eval_func_name)]

        if self.eval_func_is_higher_better:
            loss = np.array(loss)*(-1)

        best_idx_loss = np.argmin(loss)
        loss = loss[best_idx_loss]
        loss_std = loss_std[best_idx_loss]
        num_boost_round = best_idx_loss + 1

        self.num_iterations += 1
        self._print_iter()

        run_time = timer() - start

        output = {
            'loss': loss, 
            "loss_variance":loss_std**2,
            'params': model_params_, 
            "num_iterations": self.num_iterations,
            'num_boost_round': num_boost_round,
            "iteration_run_time": run_time,
        }

        self.hyperopt_summary.append(output.copy())

        output["status"] = hyperopt.STATUS_OK

        return output

    # keep track of all random_states
    def tune_hyperparams(self, 
                        model, 
                        train_valid_folds = None, 
                        max_evals = 100):

        self.hyperopt_summary = []

        bayes_trials = hyperopt.Trials()

        self.model = model
        if self.model.random_seed is not None:
            print("Overriding model's random seed of {} to {}".format(self.model.random_seed, self.random_seed))
        self.model.random_seed = self.random_seed

        if train_valid_folds is None:
            train_valid_folds =  KFold(
                                    n_splits = 5, 
                                    shuffle = True, 
                                    random_state = self.random_seed
                                )

        self._train_valid_folds = list(train_valid_folds.split(self.model.X_train))

        model_params = copy(parameter_space[self.model.model_type])

        if self.model.model_type == "random_forest":
            objective = self._objective_random_forest

        elif self.model.model_type == "elastic_net":
            # TODO: just use ElasticNetCV instead of hyperopt.fmin()

            objective = self._objective_elastic_net

        elif self.model.model_type == "lightgbm":
            self._d_train = lgb.Dataset(
                            data = self.model.X_train, 
                            label = self.model.y_train, 
                            weight = self.model.sample_weight
                        )
            
            # TODO: make lightgbm silent pls
            objective = self._objective_lightgbm
        elif self.model.model_type == "xgboost":
            self._d_train = xgb.DMatrix(
                            data = self.model.X_train, 
                            label = self.model.y_train, 
                            weight = self.model.sample_weight
                        )

            objective = self._objective_xgboost

        self.num_iterations = 0

        random_state = np.random.RandomState(self.random_seed)

        _ = hyperopt.fmin(
            fn = objective,
            space = model_params,
            algo = hyperopt.tpe.suggest,
            max_evals = max_evals,
            trials = bayes_trials,
            rstate = random_state,
        )

        self.hyperopt_summary = pd.DataFrame(self.hyperopt_summary)
        min_loss_idx = self.hyperopt_summary["loss"].idxmin()
        self.best_params = self.hyperopt_summary.loc[min_loss_idx,"params"]


    def fit_best_model(self, override_params = None):
        assert hasattr(self, "best_params"), "haven't found best model yet, need to hyperparameter tune"

        if override_params is None:
            override_params = {}

        # test override here
        model_params = copy(self.best_params)
        print(model_params)
        model_params.update(override_params)
        self.model.fit(model_params = model_params)

    # shorthand function
    def tune_and_fit(self,
                    model,
                    train_valid_folds = None,
                    max_evals = 100,
                    override_params = None):

        self.tune_hyperparams(model,
                             train_valid_folds,
                             max_evals)

        self.fit_best_model(override_params)

        return self.model