import pandas as pd
from copy import copy

from grape import model_utils
from grape.utils import prepare_folds

from xgboost import DMatrix

#todo: check xgboost sample_weights
#todo: add repr

class RegressionModel:
    """
    A regression model that can take on the following types: 
    random_forest, elastic_net, lightgbm, and xgboost

    Parameters
    ----------
    X_train : dataframe of shape [n_samples, n_features]
        The independent variables in the training set 
    y_train : list-like (numpy array or pandas series) of shape [n_samples]
        The dependent variable in the training set
    model_type : str, optional (default="random_forest")
        The underlying regression model used
        The options are: ["random_forest", "elastic_net", "lightgbm", "xgboost"]
    obj_func_name : str, optional (default="mse")
        The objective function used in training the regression model.
        The allowed objective function depends on the model_type:
            elastic_net ignores the obj_func_name since it explicity uses mse with L1 and L2 error
            random_forest has two choices: ["mse", "mae"]
            lightgbm has the following (based on lightgbm Python API):
                mse_aliases ["regression", "regression_l2", "l2", 
                            "mean_squared_error", "mse", "l2_root", 
                            "root_mean_squared_error", "rmse"]
                mae_aliases ["regression_l1", "l1", "mean_absolute_error", "mae"]
                other options ["huber", "fair", "poisson","quantile", "mape", "gamma", "tweedie"]
                objective core parameter docs https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters
            xgboost has the following options (based on xgboost Python API):
                ["mae", "mse", "reg:linear", "reg:squaredlogerror", "reg:logistic", "reg:gamma", "reg:tweedie"]
                learning task parameter docs https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
    sample_weight : list-like (numpy array or pandas series) of shape [n_samples], or None
        optional (default=None)
        Sample weights. If None, then samples are equally weighted
    random_seed : int or None, optional (default=None)
        optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.
    log_target_reg : bool, optional (default=False)
        If True, take np.log1p transform y_train before training and take np.expm1 transform during prediciton
        The advantage over passing a log transformed y_train is the model diagnostics are measured in terms of the original units
        Note: currently not available for lightgbm and xgboost
        Note: If True, sample_weight is passed directly to the underlying regression model that uses the transformed y_train
        For reference, see https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html#sklearn.compose.TransformedTargetRegressor

    Attributes
    ----------
    Attributes which are saved initialization parameters:
        X_train : dataframe of shape [n_samples, n_features]
        y_train : list-like (numpy array or pandas series) of shape [n_samples]
        model_type : str
        obj_func_name : str
        sample_weight : list-like (numpy array or pandas series) of shape [n_samples], or None
        random_seed : int, RandomState instance or None, optional (default=None)
        log_target_reg : bool

    Derived attributes:
        feature_names : list
            Feature names of independent variables

    Attributes after calling the fit method
        model_params : dict
            Parameters for each particular regression model
        model : an sklearn, xgboost, or lightgbm model
            The particular model instance, depends on the model_type

    Example
    --------
    see regression_model_test in test.py
    """

    def __init__(self, 
                   X_train,
                   y_train,
                   model_type = "random_forest", 
                   obj_func_name = "mse", 
                   sample_weight = None,
                   random_seed = None,
                   log_target_reg = False,
                ):
 
        if model_type not in ["random_forest","elastic_net", "lightgbm", "xgboost"]:
            raise NotImplementedError("{} model_type not implemented".format(model_type))

        self.X_train = X_train
        self.feature_names = X_train.columns.tolist()
        self.y_train = y_train
        self.sample_weight = sample_weight
        self.model_type = model_type
        self.obj_func_name = obj_func_name
        self.random_seed = random_seed
        self.log_target_reg = log_target_reg

        if model_type == "elastic_net":
            if sample_weight is not None:
                print("sample_weight ignored for elastic_net")
            if obj_func_name is not None:
                print("obj_func_name is ignored for elastic_net")


    def fit(self, model_params = None):
        """
        Fits a regression model using training data

        Parameters
        ----------
        model_params : dict or None
            If dict, contains parameters specific to each regression model
            If None, will take on default parameters of the respective regression models
            See the following links for documentation:
                sklearn elastic_net docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
                sklearn random_forest docs: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
                lightgbm docs: https://lightgbm.readthedocs.io/en/latest/Parameters.html
                xgboost docs: https://xgboost.readthedocs.io/en/latest/parameter.html
             Note: see _prepare_params method for the parameter default overrides per regression model type
        """

        self._prepare_params(model_params)

        if self.model_type == "elastic_net":
            self.model = model_utils._fit_elastic_net(self.X_train, self.y_train, 
                                                        self.model_params, self.log_target_reg)

        elif self.model_type == "random_forest":
            self.model = model_utils._fit_random_forest(self.X_train, self.y_train, 
                                                        self.obj_func_name, self.sample_weight, 
                                                        self.model_params, self.log_target_reg)

        elif self.model_type == "lightgbm":
            self.model = model_utils._fit_lightgbm(self.X_train, self.y_train, 
                                                    self.obj_func_name, self.sample_weight, 
                                                    self.model_params, self.log_target_reg)

        elif self.model_type == "xgboost":
            self.model = model_utils._fit_xgboost(self.X_train, self.y_train, 
                                                    self.obj_func_name, self.sample_weight, 
                                                    self.model_params, self.log_target_reg)

    def cross_validate(self, 
                        train_valid_folds = 5,
                        eval_func_names = "mse", 
                        model_params = None,
                        include_rf_oob = True):
        """
        Computes cross validation scores based on evaluation function/s and train-valid folds

        Parameters
        ----------
        train_valid_folds : int, list of lists of ints, or cross validation generator 
            Optional (default=5)
            The folds used for cross validation
            If int, the number of folds used in sklearn's KFold 
            If list of list of indices, each sublist should contain the indices per fold
        eval_func_names : str of list of str, optional (default="mse")
            The evaluation function used to measure cross validation error
            The options are: ["r_squared", "mse", "rmse", "mae"]
        model_params : dict or None
            If dict, contains parameters specific to each regression model
            If None, will take on default parameters of the respective regression models
            See the following links for documentation:
                sklearn elastic_net docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
                sklearn random_forest docs: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
                lightgbm docs: https://lightgbm.readthedocs.io/en/latest/Parameters.html
                xgboost docs: https://xgboost.readthedocs.io/en/latest/parameter.html
            Note: see _prepare_params method for the parameter default overrides per regression model type
        include_rf_oob : bool, optional (default=True)
            Whether or not to include oob error in cv_scores
            Only applicable when model_type = "random_forest"
            On a model_type not "random_forest", this is ignored

        Returns 
        -------
        cv_scores : dict
            Cross validation error summary
            Contains the following, per evaluation function:
                cv error averaged across folds
                cv error's standard deviation across folds
                out-of-bag error, when available
            Example: {"cv-mse-mean":34.13, "cv-mse-std":2.67, "oob-mse":33.2}
        """
        
                        
        if isinstance(eval_func_names, str):
            eval_func_names = [eval_func_names]

        train_valid_folds = prepare_folds(train_valid_folds,
                                            self.X_train,
                                            self.random_seed)

        self._prepare_params(model_params)

        if self.model_type == "elastic_net":
            cv_scores = model_utils._cv_elastic_net(self.X_train, self.y_train, train_valid_folds, 
                                                    eval_func_names,
                                                    model_params,
                                                    self.log_target_reg)

        elif self.model_type == "random_forest":
            if include_rf_oob:
                assert hasattr(self, "model"), "random_forest must be trained first to include oob error"
                oob_pred = self.model.oob_prediction_
            else:
                oob_pred = None
            cv_scores = model_utils._cv_random_forest(self.X_train, self.y_train, train_valid_folds,
                                                    self.obj_func_name, 
                                                    eval_func_names,
                                                    model_params,
                                                    self.sample_weight,
                                                    self.log_target_reg,
                                                    oob_pred)


        elif self.model_type == "lightgbm":
            cv_scores = model_utils._cv_lightgbm(self.X_train, self.y_train, train_valid_folds, 
                                                self.obj_func_name,
                                                eval_func_names,
                                                model_params,
                                                self.sample_weight,
                                                self.log_target_reg)
           

        elif self.model_type == "xgboost":
            cv_scores = model_utils._cv_xgboost(self.X_train, self.y_train, train_valid_folds, 
                                                self.obj_func_name,
                                                eval_func_names,
                                                model_params,
                                                self.sample_weight,
                                                self.log_target_reg)
            
        else:
            raise NotImplementedError("model type {} not supported".format(self.model_type))

        return cv_scores

    def predict(self, X_pred):
        """
        Computes regression predictions

        Parameters
        ----------
        X_pred : array-like of shape [n_samples_pred, n_features]

        Returns 
        -------
        predictions : numpy array of shape [n_samples_pred]
        """
        if self.model_type == "xgboost":
            y_pred = model_utils._xgb_predict(self.model, X_pred)
        else:
            y_pred = self.model.predict(X_pred)
        
        return y_pred

    def _prepare_params(self, model_params):
        """
        Helper function for setting parameter defaults per regression model type
        """

        if model_params is None:
            self.model_params = {}
        else:
            assert "random_state" not in model_params.keys(), "random_state should not be explicitly set within the model_params dictionary, random_seed should be set in the RegressionModel constructor instead"
            assert "seed" not in model_params.keys(), "seed should not be explicitly set within the model_params dictionary, random_seed should be set in the RegressionModel constructor instead"
            self.model_params = copy(model_params)

        if self.model_type == "elastic_net":
            self.model_params["max_iter"] = 10000
            self.model_params["random_state"] = self.random_seed
        

        elif self.model_type == "random_forest":
            self.model_params["n_jobs"] = -1
            self.model_params["oob_score"] = True

            if "n_estimators" not in self.model_params.keys():
                self.model_params["n_estimators"] = 100

            self.model_params["random_state"] = self.random_seed

        elif self.model_type == "lightgbm":
            self.model_params["n_jobs"] = -1
            self.model_params["random_state"] = self.random_seed

        elif self.model_type == "xgboost":
            self.model_params["verbosity"] = 1

            if self.sample_weight is not None:
                print("Sample weight not yet supported with the XGBoost model")
                self.sample_weight = None
            
            if self.random_seed is None:
                self.model_params["seed"] = 0
            else:
                self.model_params["seed"] = self.random_seed

        else:
            raise NotImplementedError("model type {} not supported".format(self.model_type))

 