import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from copy import copy

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV

from utils import _huber_approx_obj

from lightgbm.basic import Booster as LightBooster
from xgboost.core import Booster as XgBooster

class RegressionModel:

    def __init__(self, 
                   model_type, 
                   metric = "l2", 
                   model_params = None, 
                   verbosity = 0,
                   sample_weight = None,
                   custom_fobj = None, 
                   custom_feval = None,
                   prepare_diagnostics = True
                ):
        '''
        Positional Arguements:
            model_type
        Keyword Arguements:
            metric: String. Optional. 'l1' or 'l2'.
        '''

        assert model_type in ["random_forest","elastic_net", "lightgbm", "lightgbm_special", "xgboost"]

        self.verbosity = verbosity
        self.model_type = model_type
        self.custom_fobj = custom_fobj
        self.custom_feval = custom_feval
        self.prepare_diagnostics = prepare_diagnostics

        # NOTE: do this incase it is written differently in other implementations
        if model_type in ["random_forest", "elastic_net", "xgboost"]:
            if metric == "l1":
                self._metric = "mae"
            elif metric == "l2":
                self._metric = "mse"
        elif model_type == "lightgbm":
            self._metric = "regression_{}".format(metric)
        elif model_type == "lightgbm_special":
            self._metric = metric

        param_key = model_type
        if model_type == "lightgbm_special":
            param_key = "lightgbm"

        if model_params is None:
            self._model_params = {}
        else:
            self._model_params = model_params

    @classmethod
    def from_trained(cls, trained_model, **kwargs):
        if isinstance(trained_model, RandomForestRegressor):
            model_string = "random_forest"
        elif isinstance(trained_model, ElasticNetCV) or isinstance(trained_model, ElasticNet):
            model_string = "elastic_net"
        elif isinstance(trained_model, LightBooster):
            model_string = "lightgbm"
        elif isinstance(trained_model, XgBooster):
            model_string = "xgboost"
        else:
            return None
        
        regression_model = cls(model_string, **kwargs)
        regression_model.model = trained_model

        return regression_model

    def fit(self, X_train, y_train, 
            sample_weight = None):
        '''
        Positional Arguements:
            X_train
            y_train
        Keyword Arguements:
            sample_weight
            train_valid_folds should be of type splitter like KFold
        '''
        self.feature_names = X_train.columns.tolist()
        self.sample_weight = sample_weight

        if self.prepare_diagnostics:
            self.X_train = X_train
            self.y_train = y_train

        # TODO: raise a more appropriate exception (NotImplemented)
        if (self.custom_fobj is not None) or (self.custom_feval is not None):
            assert self.model_type == "lightgbm_special", "custom_obj or custom_feval only for lightgbm_special"

        if self.model_type == "elastic_net":
            self._fit_elastic_net(X_train, y_train)

        elif self.model_type == "random_forest":
            self._fit_random_forest(X_train, y_train)

        elif self.model_type in ["lightgbm", "lightgbm_special"]:
            self._fit_lightgbm(X_train, y_train)

        elif self.model_type == "xgboost":
            self._fit_xgboost(X_train, y_train)

    def _fit_elastic_net(self, X_train, y_train):

        self.model = ElasticNet(**self._model_params)
        
        self.model.fit(X = X_train, y = y_train)

    def _fit_random_forest(self, X_train, y_train):
        self._model_params["n_jobs"] = -1
        self._model_params["criterion"] = self._metric

        self.model = RandomForestRegressor(**self._model_params)
        self.model.fit(X = X_train, y = y_train, 
                        sample_weight = self.sample_weight)

    def _fit_lightgbm(self, X_train, y_train):
        self._model_params["n_jobs"] = -1

        if self.model_type == "lightgbm":
            self._model_params["objective"] = self._metric
            
        if self.model_type == "lightgbm_special":
            if self.verbosity >= 1:
                print("Ignoring the set metric {} for lightgbm_special".format(self._metric))
            if sample_weight is not None:
                if self.verbosity >= 1:
                    print("Disregarding sample_weights (for mape or mspe) for lightgbm_special")
                sample_weight = None
            self._model_params["metric"] = "None"
        
        d_train = lgb.Dataset(data = X_train, label = y_train, weight = self.sample_weight)
            
        self.model = lgb.train(params = self._model_params, 
                                train_set = d_train, 
                                fobj = self.fobj, 
                                feval = self.feval)

    def _fit_xgboost(self, X_train, y_train):
        self._model_params["silent"] = 1

        if self._metric == "mae":
            # NOTE: this probably doesn't work yet
            obj = self._huber_approx_obj
        if self._metric == "mse":
            obj = None

        if sample_weight is not None:
            print("Sample weight not yet supported with the XGBoost model")
            sample_weight = None

        d_train = xgb.DMatrix(data = X_train, label = y_train, 
                                weight = self.sample_weight)

        self.model = xgb.train(params = self._model_params, 
                                dtrain = d_train, 
                                obj=obj, 
                                num_boost_round = self._model_params.get("num_boost_round", 200))

    def predict(self, X_):
        if self.model_type == "xgboost":
            X_pred = xgb.DMatrix(X_)
        else:
            X_pred = copy(X_)
        
        predictions = self.model.predict(X_pred)
        
        return predictions
