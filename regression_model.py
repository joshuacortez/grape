import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from copy import copy

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import _huber_approx_obj

from lightgbm.basic import Booster as LightBooster
from xgboost.core import Booster as XgBooster

class RegressionModel:

    def __init__(self, 
                   X_train,
                   y_train,
                   model_type, 
                   metric = "l2", 
                   verbosity = 0,
                   sample_weight = None,
                   custom_fobj = None, 
                   custom_feval = None,
                ):
        '''
        Positional Arguements:
            X_train
            y_train
            model_type
        Keyword Arguements:
            sample_weight
            metric: String. Optional. 'l1' or 'l2'.
        '''
        assert model_type in ["random_forest","elastic_net", "lightgbm", "lightgbm_special", "xgboost"]
        # TODO: raise a more appropriate exception (NotImplemented)
        if (custom_fobj is not None) or (custom_feval is not None):
            assert model_type == "lightgbm_special", "custom_obj or custom_feval only for lightgbm_special"

        self.X_train = X_train
        self.feature_names = X_train.columns.tolist()
        self.y_train = y_train
        self.sample_weight = sample_weight
        self.verbosity = verbosity
        self.model_type = model_type
        self.custom_fobj = custom_fobj
        self.custom_feval = custom_feval

        # NOTE: do this incase it is written differently in other implementations
        if model_type in ["random_forest", "elastic_net", "xgboost"]:
            if metric == "l1":
                self.metric = "mae"
            elif metric == "l2":
                self.metric = "mse"
        elif model_type == "lightgbm":
            self.metric = "regression_{}".format(metric)
        elif model_type == "lightgbm_special":
            self.metric = metric

    def fit(self, model_params = None):
        if model_params is None:
            self.model_params = {}
        else:
            self.model_params = model_params

        if self.model_type == "elastic_net":
            self._prepare_elastic_net_params()
            self.model = self._fit_elastic_net(self.X_train, self.y_train, self.model_params)

        elif self.model_type == "random_forest":
            self._prepare_random_forest_params()
            self.model = self._fit_random_forest(self.X_train, self.y_train, self.sample_weight, self.model_params)

        elif self.model_type in ["lightgbm", "lightgbm_special"]:

            self._prepare_lightgbm_params()
            self.model = self._fit_lightgbm(self.X_train, self.y_train, self.sample_weight, self.model_params, 
                                            # todo: two arguments below being inside self.model_params
                                            self.custom_fobj,
                                            self.custom_feval)

        elif self.model_type == "xgboost":
            self._prepare_xgboost_params()
            self.model = self._fit_xgboost(self.X_train, self.y_train, self.sample_weight, self.model_params, 
                                            # todo: argument below being inside self.model_params
                                            self.metric)

    def _prepare_elastic_net_params(self):
        self.model_params["max_iter"] = 10000

    @staticmethod
    def _fit_elastic_net(X_train, y_train, model_params = None):
        if model_params is None:
            model_params = {}
        model = ElasticNet(**model_params)
        pipeline = make_pipeline(StandardScaler(), model)
        pipeline.fit(X = X_train, y = y_train)

        return pipeline

    def _prepare_random_forest_params(self):
        self.model_params["n_jobs"] = -1
        self.model_params["criterion"] = self.metric
        self.model_params["oob_score"] = True

        if "n_estimators" not in self.model_params.keys():
            self.model_params["n_estimators"] = 100

    @staticmethod
    def _fit_random_forest(X_train, y_train, sample_weight = None, model_params = None):
        if model_params is None:
            model_params = {}

        model = RandomForestRegressor(**model_params)
        model.fit(X = X_train, y = y_train, sample_weight = sample_weight)
        return model

    def _prepare_lightgbm_params(self):
        self.model_params["n_jobs"] = -1

        if self.model_type == "lightgbm":
            self.model_params["objective"] = self.metric
            
        if self.model_type == "lightgbm_special":
            if self.verbosity >= 1:
                print("Ignoring the set metric {} for lightgbm_special".format(self.metric))
            if self.sample_weight is not None:
                if self.verbosity >= 1:
                    print("Disregarding sample_weights (for mape or mspe) for lightgbm_special")
                self.sample_weight = None

        self.model_params["metric"] = "None"

    @staticmethod
    def _fit_lightgbm(X_train, y_train, sample_weight = None, model_params = None, 
                        custom_fobj = None, custom_feval = None):     
        if model_params is None:
            model_params = {}

        d_train = lgb.Dataset(data = X_train, label = y_train, weight = sample_weight)
            
        model = lgb.train(params = model_params, 
                          train_set = d_train, 
                          fobj = custom_fobj, 
                          feval = custom_feval)

        return model

    def _prepare_xgboost_params(self):
        self.model_params["silent"] = 1

        if self.sample_weight is not None:
            print("Sample weight not yet supported with the XGBoost model")
            self.sample_weight = None

    @staticmethod
    def _fit_xgboost(X_train, y_train, sample_weight = None, model_params = None, 
                        metric = "mse"):
        if model_params is None:
            model_params = {}

        if metric == "mae":
            # NOTE: this probably doesn't work yet
            obj = _huber_approx_obj
        if metric == "mse":
            obj = None

        d_train = xgb.DMatrix(data = X_train, label = y_train, 
                                weight = sample_weight)

        model = xgb.train(params = model_params, 
                          dtrain = d_train, 
                          obj=obj, 
                          num_boost_round = model_params.get("num_boost_round", 200))

        return model

    def predict(self, X_pred):
        if self.model_type == "xgboost":
            X_pred = xgb.DMatrix(X_pred)
        #else:
        #    X_pred = copy(X_pred)
        
        predictions = self.model.predict(X_pred)
        
        return predictions
