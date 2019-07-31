import pandas as pd
from copy import copy

import model_utils

from xgboost import DMatrix

#todo: check xgboost sample_weights

class RegressionModel:

    def __init__(self, 
                   X_train,
                   y_train,
                   model_type = "random_forest", 
                   obj_func_name = "mse", 
                   verbosity = 0,
                   sample_weight = None,
                   random_seed = None
                ):
        '''
        Positional Arguements:
            X_train
            y_train
            model_type
        Keyword Arguements:
            sample_weight
            obj_func_name: String. Optional. 'mae' or 'mse'.
        '''
        assert model_type in ["random_forest","elastic_net", "lightgbm", "xgboost"]
        # TODO: raise a more appropriate exception (NotImplemented)

        self.X_train = X_train
        self.feature_names = X_train.columns.tolist()
        self.y_train = y_train
        self.sample_weight = sample_weight
        self.verbosity = verbosity
        self.model_type = model_type
        self.obj_func_name = obj_func_name
        self.random_seed = random_seed

        if (model_type == "elastic_net") and (sample_weight is not None):
            print("sample_weight ignored for elastic_net")

    def fit(self, model_params = None):
        if model_params is None:
            self.model_params = {}
        else:
            self.model_params = model_params

        if self.model_type == "elastic_net":
            self._prepare_elastic_net_params()
            self.model = model_utils._fit_elastic_net(self.X_train, self.y_train, 
                                                        self.model_params)

        elif self.model_type == "random_forest":
            self._prepare_random_forest_params()
            self.model = model_utils._fit_random_forest(self.X_train, self.y_train, 
                                                        self.obj_func_name, self.sample_weight, self.model_params)

        elif self.model_type == "lightgbm":

            self._prepare_lightgbm_params()
            self.model = model_utils._fit_lightgbm(self.X_train, self.y_train, 
                                                    self.obj_func_name, self.sample_weight, self.model_params)

        elif self.model_type == "xgboost":
            self._prepare_xgboost_params()
            self.model = model_utils._fit_xgboost(self.X_train, self.y_train, 
                                                    self.obj_func_name, self.sample_weight, self.model_params)

    def cross_validate(self, 
                        train_valid_folds,
                        eval_func_names = None, 
                        model_params = None):
                        
        if model_params is None:
            self.model_params = {}
        else:
            assert "random_state" not in model_params.keys(), "seed should not be explicitly set within the model_params dictionary"
            assert "seed" not in model_params.keys(), "seed should not be explicitly set within the model_params dictionary"
            self.model_params = model_params
        if eval_func_names is None:
            eval_func_names = ["mse"]
        if isinstance(eval_func_names, str):
            eval_func_names = [eval_func_names]

        if not isinstance(train_valid_folds, list):
            train_valid_folds = list(train_valid_folds.split(self.X_train))

        if model_params is None:
            assert hasattr(self, "model_params"), "If model hasn't been fitted, must supply model_params"

        if self.model_type == "elastic_net":
            self._prepare_elastic_net_params()
            cv_scores = model_utils._cv_elastic_net(self.X_train, self.y_train, train_valid_folds, 
                                                    eval_func_names,
                                                    model_params)

        elif self.model_type == "random_forest":
            self._prepare_random_forest_params()
            if hasattr(self, "model"):
                if hasattr(self.model, "oob_prediction_"):
                    oob_prediction = self.model.oob_prediction_
                else:
                    oob_prediction = None
            else:
                oob_prediction = None
            cv_scores = model_utils._cv_random_forest(self.X_train, self.y_train, train_valid_folds,
                                                    self.obj_func_name, 
                                                    eval_func_names,
                                                    model_params,
                                                    self.sample_weight,
                                                    oob_prediction)


        elif self.model_type == "lightgbm":
            # still not working for custom objective function
            self._prepare_lightgbm_params()
            cv_scores = model_utils._cv_lightgbm(self.X_train, self.y_train, train_valid_folds, 
                                                self.obj_func_name,
                                                eval_func_names,
                                                model_params,
                                                self.sample_weight)
           

        elif self.model_type == "xgboost":
            self._prepare_xgboost_params()
            cv_scores = model_utils._cv_xgboost(self.X_train, self.y_train, train_valid_folds, 
                                                self.obj_func_name,
                                                eval_func_names,
                                                model_params,
                                                self.sample_weight)
            
        else:
            raise Exception("model type {} not supported".format(self.model_type))

        return cv_scores


    def _prepare_elastic_net_params(self):
        self.model_params["max_iter"] = 10000
        self.model_params["random_state"] = self.random_seed

    def _prepare_random_forest_params(self):
        self.model_params["n_jobs"] = -1
        self.model_params["oob_score"] = True

        if "n_estimators" not in self.model_params.keys():
            self.model_params["n_estimators"] = 100

        self.model_params["random_state"] = self.random_seed

    def _prepare_lightgbm_params(self):
        self.model_params["n_jobs"] = -1

        self.model_params["random_state"] = self.random_seed

    def _prepare_xgboost_params(self):
        self.model_params["silent"] = 1

        if self.sample_weight is not None:
            print("Sample weight not yet supported with the XGBoost model")
            self.sample_weight = None
        
        if self.random_seed is None:
            self.model_params["seed"] = 0
        else:
            self.model_params["seed"] = self.random_seed


    def predict(self, X_pred):
        if self.model_type == "xgboost":
            X_pred = DMatrix(X_pred)
        #else:
        #    X_pred = copy(X_pred)
        
        predictions = self.model.predict(X_pred)
        
        return predictions

 