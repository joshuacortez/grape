import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV

import hyperopt

import numpy as np
from numpy.random import RandomState
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from timeit import default_timer as timer
from copy import copy

from regression_model import RegressionModel
from param_space import parameter_space, int_params_dict
from utils import get_elastic_net_l1_ratio, _huber_approx_obj, eval_func_is_higher_better, prepare_folds


# todo: hyperparameter optimization that depends on time, or can stop midway

class HyperParameterOptimizer:
    """
    Hands-free hyperparameter optimization powered by Hyperopt
    Traditional methods use grid search and random search. 
    However, this instead uses Sequential Model Based Optimization (SMBO)
    In SMBO, trials of hyperparams are informed by earlier trials of hyperparams

    Parameters
    ----------
    eval_func_name : str, optional (default="mse")
        The evaluation function used to measure cross validation error
        The hyperparameters that maximize/minimize the evaluation function
        are chosen
    verbosity : int, optional (default=0)
        If verbosity >= 2, print after every iteration

    Attributes
    ----------
    Attributes which are saved initialization parameters:
        eval_func_name : str
        verbosity : int
    
    Derived attributes after calling tune_hyperparams or tune_and_fit:
        hyperopt_summary : dataframe of shape [total_n_iterations, 5]
            Contains the following information per iteration of HPO:
                loss : mean of cross validation error
                loss_variance : variance of cross validation error
                params : hyperparameters used in iteration
                n_iterations : counter variable for the number of iterations
                iteration_run_time : runtime in seconds
        best_params : dict
            Contains the hyperparameters under the minimum loss
        model : a GRAPE RegresssionModel
            Tuned with the best hyperparameters

    Example
    -------
    see hyperparameter_optimizer_test in tests.py

    References
    ----------
    https://github.com/WillKoehrsen/hyperparameter-optimization/blob/master/Bayesian%20Hyperparameter%20Optimization%20of%20Gradient%20Boosting%20Machine.ipynb
    """

    def __init__(self, eval_func_name = "mse", verbosity = 0):
        self.verbosity = verbosity
        self.eval_func_name = eval_func_name

    # keep track of all random_states
    def tune_hyperparams(self, 
                        model, 
                        train_valid_folds = 5, 
                        total_n_iterations = 100,
                        use_model_copy = False):
        """
        Finds the best hyperparameters for a GRAPE RegressionModel

        Parameters
        ----------
        model : a GRAPE RegressionModel
        train_valid_folds : int, list of lists of ints, or cross validation generator 
            Optional (default=5)
            The folds used for cross validation
            If int, the number of folds used in sklearn's KFold 
            If list of list of indices, each sublist should contain the indices per fold
        total_n_iterations : int
            The number of iterations to run hyperparameter optimization until it stops
            The hyperparameter with lowest loss within total_n_iterations is chosen
        use_model_copy : bool
            If False, pass only a reference to the GRAPE RegressionModel, saving memory
            If True, makes a copy of the model, ensuring the original instance doesn't get overwritten
        """

        self.hyperopt_summary = []

        bayes_trials = hyperopt.Trials()

        if use_model_copy:
            model = copy(model)
        self.model = model

        train_valid_folds = prepare_folds(train_valid_folds,
                                            self.model.X_train,
                                            self.model.random_seed)
        self._train_valid_folds = train_valid_folds

        model_params = copy(parameter_space[self.model.model_type])

        if self.model.model_type == "lightgbm":
            self._d_train = lgb.Dataset(
                            data = self.model.X_train, 
                            label = self.model.y_train, 
                            weight = self.model.sample_weight
                        )
            
            # TODO: make lightgbm silent pls
        elif self.model.model_type == "xgboost":
            self._d_train = xgb.DMatrix(
                            data = self.model.X_train, 
                            label = self.model.y_train, 
                            weight = self.model.sample_weight
                        )

        self._n_iterations = 0

        random_state = RandomState(self.model.random_seed)

        _ = hyperopt.fmin(
            fn = self._objective_callback,
            space = model_params,
            algo = hyperopt.tpe.suggest,
            max_evals = total_n_iterations,
            trials = bayes_trials,
            rstate = random_state,
        )

        self.hyperopt_summary = pd.DataFrame(self.hyperopt_summary)
        min_loss_idx = self.hyperopt_summary["loss"].idxmin()
        self.best_params = self.hyperopt_summary.loc[min_loss_idx,"params"]
       
    def fit_best_model(self, override_params = None):
        """
        Fits a GRAPE RegressionModel with the best found hyperparameters

        Parameters
        ----------
        override_params : dict or None, optional (default=None)
            If dict, contains parameters to add to best_params or to override 
            Example why this could be used: 
            In random_forest, higher values n_estimators (i.e. number of decision trees) 
            can only make predictive performance better. But it comes at the 
            expense of computational time. During hyperparameter tuning, 
            it's wise to keep it at a relatively low number (e.g. n_estimators = 100) 
            for good runtime. After tuning, it's a good idea to increase n_estimators 
            (e.g. n_estimators = 400) to increase predictive performance 
        """

        assert hasattr(self, "best_params"), "Need to tune_hyperparams first"

        if override_params is None:
            override_params = {}

        model_params = copy(self.best_params)
        print(model_params)
        model_params.update(override_params)
        self.model.fit(model_params = model_params)

    # shorthand function
    def tune_and_fit(self,
                    model,
                    train_valid_folds = None,
                    total_n_iterations = 100,
                    use_model_copy = False,
                    override_params = None):
        """
        Convenience function for both finding and fitting the best hyperparams

        Parameters
        ----------
        see documentation of tune_hyperparams and fit_best_model
        """

        self.tune_hyperparams(model,
                             train_valid_folds,
                             total_n_iterations,
                             use_model_copy)

        self.fit_best_model(override_params)
    
    def _objective_callback(self, model_params):
        """
        Callback function that Hyperopt's fmin function attempts to minimize
        More specifically, this computes the cross validation error of the model
        Hyperopt's fmin function is incentivized to minimize cross validation error
        """
        start_time = timer()
        model_params = copy(model_params)
        model_params = self._prepare_params(model_params, 
                                            model_type = self.model.model_type)

        cv_scores = self.model.cross_validate(self._train_valid_folds,
                                             self.eval_func_name,
                                             model_params,
                                             include_rf_oob = False)

        loss = cv_scores["cv-{}-mean".format(self.eval_func_name)]
        loss_std = cv_scores["cv-{}-std".format(self.eval_func_name)]
        if eval_func_is_higher_better[self.eval_func_name]:
            loss = loss*(-1)

        # xgboost and lightgbm have the additional intricacy of having boosting rounds
        if self.model.model_type in ["xgboost", "lightgbm"]:
            best_idx_loss = np.argmin(loss)
            loss = loss[best_idx_loss]
            loss_std = loss_std[best_idx_loss]
            num_boost_round = best_idx_loss + 1
            
            if self.model.model_type == "lightgbm":
                model_params["estimators"] = num_boost_round
            elif self.model.model_type == "xgboost":
                model_params["num_boost_round"] = num_boost_round

        self._n_iterations += 1

        self._print_iter()

        run_time = timer() - start_time

        output = {
            'loss': loss, 
            'loss_variance':loss_std**2,
            'params': model_params, 
            "n_iterations": self._n_iterations, 
            "iteration_run_time": run_time,
        }

        self.hyperopt_summary.append(output.copy())

        output["status"] = hyperopt.STATUS_OK

        return output

    def _print_iter(self):
        if self.verbosity >= 2:
            print("Iteration:", self._n_iterations)

    @staticmethod
    def _prepare_params(model_params, model_type):
        """
        Convenience function for conditional sampling (if applicable),
        and for type conversions (converting to int) of Hyperopt's parameters
        """
        if model_type == "elastic_net":
            model_params["l1_ratio"] = get_elastic_net_l1_ratio(model_params)
        if model_type == "lightgbm":
            # conditional sampling from bayesian domain for the goss bossting type
            if "boosting_type" in model_params.keys():
                subsample = model_params['boosting_type'].get('subsample', 1.0)

                model_params['boosting_type'] = model_params['boosting_type']['boosting_type']
                model_params['subsample'] = subsample

        # converting to int
        for param in int_params_dict[model_type]:
            if param in model_params.keys():
                model_params[param] = int(model_params[param])
        
        return model_params