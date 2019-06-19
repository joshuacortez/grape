import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from copy import copy

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV

from param_space import parameter_space
from utils import _huber_approx_obj

from lightgbm.basic import Booster as LightBooster
from xgboost.core import Booster as XgBooster

class RegressionModel:

    def __init__(self, model_type, **kwargs):
        '''
        Positional Arguements:
            model_type
        Keyword Arguements:
            metric: String. Optional. 'l1' or 'l2'.
        '''

        assert model_type in ["random_forest","elastic_net", "lightgbm", "lightgbm_special", "xgboost"]

        self.verbosity = kwargs.get("verbosity", 0)
        metric = kwargs.get("metric", "l2")

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

        self._model_type = model_type

        param_key = model_type
        if model_type == "lightgbm_special":
            param_key = "lightgbm"
        self._param_space = copy(parameter_space[param_key])

    @classmethod
    def from_trained(cls, trained_model, **kwargs):
        feature_names = kwargs.get("feature_names", None)

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

        if model_string == "random_forest":
            regression_model.feature_importance_df = pd.DataFrame(
                data = {"feature_importance": trained_model.feature_importances_}, 
                index = feature_names
            )
        elif model_string == "elastic_net":
            regression_model.feature_importance_df = pd.DataFrame(
                data = {"feature_importance": trained_model.coef_}, 
                index = feature_names
            )
        elif model_string == "light_gbm":
            regression_model.feature_importance_df = pd.DataFrame(
                data = {"feature_importance": trained_model.feature_importance()}, 
                index = feature_names
            )
        elif model_string == "xgboost":
            regression_model.feature_importance_df = pd.DataFrame.from_dict(
                {"feature_importance": trained_model.get_score(importance_type = "gain")}
            )

        return regression_model

    def fit(self, X_train, y_train, **kwargs):
        '''
        Positional Arguements:
            X_train
            y_train
        Keyword Arguements:
            sample_weight
            train_valid_folds
        '''
        sample_weight = kwargs.get("sample_weight", None)
        train_valid_folds = kwargs.get("train_valid_folds", None)

        # TODO: raise a more appropriate exception (NotImplemented)
        if train_valid_folds is not None:
            assert self._model_type == "elastic_net", "Auto CV (non-bayesian opt) only implemented for elastic_net"

        if self._model_type == "elastic_net":
            self._fit_elastic_net(X_train, y_train, **kwargs)
        elif self._model_type == "random_forest":
            self._fit_random_forest(X_train, y_train, **kwargs)
        elif self._model_type in ["lightgbm", "lightgbm_special"]:
            self._fit_lightgbm(X_train, y_train, **kwargs)
        elif self._model_type == "xgboost":
            self._fit_xgboost(X_train, y_train, **kwargs)

    def _fit_elastic_net(self, X_train, y_train, **kwargs):
        sample_weight = kwargs.get("sample_weight", None)
        train_valid_folds = kwargs.get("train_valid_folds", None)

        if train_valid_folds is not None:
            self._param_space["n_jobs"] = -1
            cv = train_valid_folds.split(X_train)
            self.model = ElasticNetCV(**self._param_space, cv = cv)
        else:
            self.model = ElasticNet(**self._param_space)
        
        self.model.fit(X = X_train, y = y_train)
        self.feature_importance_df = pd.DataFrame(data = {"feature_importance":model.coef_}, index = X_train.columns)

    def _fit_random_forest(self, X_train, y_train, **kwargs):
        sample_weight = kwargs.get("sample_weight", None)
        train_valid_folds = kwargs.get("train_valid_folds", None)

        self._param_space["n_jobs"] = -1
        self._param_space["criterion"] = self._metric

        self.model = RandomForestRegressor(**self._param_space)
        self.model.fit(X = X_train, y = y_train, sample_weight = sample_weight)

        self.feature_importance_df = pd.DataFrame(data = {"feature_importance":model.feature_importances_}, 
            index = X_train.columns)

    def _fit_lightgbm(self, X_train, y_train, **kwargs):
        sample_weight = kwargs.get("sample_weight", None)
        train_valid_folds = kwargs.get("train_valid_folds", None)
        custom_obj = kwargs.get("custom_obj", None)
        custom_feval = kwargs.get("custom_feval", None)

        self._param_space["n_jobs"] = -1

        if self._model_type == "lightgbm":
            self._param_space["objective"] = self._metric
            fobj = None
            feval = None
        if self._model_type == "lightgbm_special":
            if self.verbosity >= 1:
                print("Ignoring the set metric {} for lightgbm_special".format(self._metric))
            if sample_weight is not None:
                if self.verbosity >= 1:
                    print("Disregarding sample_weights (for mape or mspe) for lightgbm_special")
                sample_weight = None
            self._param_space["metric"] = "None"

            fobj = custom_obj
            feval = custom_feval
        
        d_train = lgb.Dataset(data = X_train, label = y_train, weight = sample_weight)
            
        self.model = lgb.train(params = self._param_space, train_set = d_train, 
            fobj = fobj, feval = feval)
        self.feature_importance_df = pd.DataFrame(data = {"feature_importance":model.feature_importance()}, 
            index = X_train.columns)

    def _fit_xgboost(self, X_train, y_train, **kwargs):
        sample_weight = kwargs.get("sample_weight", None)
        train_valid_folds = kwargs.get("train_valid_folds", None)

        self._param_space["silent"] = 1

        if self._metric == "mae":
            # NOTE: this probably doesn't work yet
            obj = self._huber_approx_obj
        if self._metric == "mse":
            obj = None

        if sample_weight is not None:
            print("Sample weight not yet supported with the XGBoost model")
            sample_weight = None

        d_train = xgb.DMatrix(data = X_train, label = y_train, weight = sample_weight)

        self.model = xgb.train(params = self._param_space, dtrain = d_train, 
            obj=obj, num_boost_round = self._param_space.get("num_boost_round", 200))
        self.feature_importance_df = pd.DataFrame.from_dict({"feature_importance":model.get_score(importance_type = "gain")})

    def predict(self, X_):
        if self._model_type == "xgboost":
            X_pred = xgb.DMatrix(X_)
        else:
            X_pred = copy(X_)
        
        predictions = self.model.predict(X_pred)
        
        return predictions
