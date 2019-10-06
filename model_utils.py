import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.compose import TransformedTargetRegressor

from utils import _huber_approx_obj, eval_func_all_dict, xgb_eval_func_consolidator

import xgboost as xgb
import lightgbm as lgb

from functools import partial

# global parameter for xgboost and lightgbm 
# set to False for not because early stopping still isn't implemented in xgboost's and lightgbm's sklearn API
USE_SKLEARN_API = False

# need to check how get_model_weights is really used

def _check_obj_func_name(obj_func_name, model_type):
    if model_type == "random_forest":
        allowed_list = ["mae", "mse"]
    elif model_type == "lightgbm":
        mse_aliases = ["regression", "regression_l2", "l2", "mean_squared_error", "mse", "l2_root", "root_mean_squared_error", "rmse"]
        mae_aliases = ["regression_l1", "l1", "mean_absolute_error", "mae"]
        allowed_list = ["huber", "fair", "poisson","quantile", "mape", "gamma", "tweedie"]
        allowed_list = allowed_list + mse_aliases + mae_aliases
    elif model_type == "xgboost":
        allowed_list = ["mae", "mse", "reg:linear", "reg:squaredlogerror", "reg:logistic", "reg:gamma", "reg:tweedie"]
    else:
        raise NotImplementedError("model type {} not supported".format(model_type))

    assert obj_func_name in allowed_list, "{} has only following objective functions available: {}".format(model_type, obj_func_name)

def _log_target_regressor(regressor, func = np.log1p, inverse_func = np.expm1):
    new_regressor = TransformedTargetRegressor(regressor, func = func, inverse_func = inverse_func)

    return new_regressor

def _cv_score_summary(cv_scores):
    """
    Helper function for summarizing cv error 
    Computes mean and standard deviation of cv error across folds
    """
    output = {}
    for score_name, scores in cv_scores.items():
        if "test" not in score_name:
            continue
        else:
            score_name = score_name.replace("test_", "cv-")
            output["{}-mean".format(score_name)] = np.mean(scores)
            output["{}-std".format(score_name)] = np.std(scores)

    return output

def _fit_elastic_net(X_train, y_train,
                    model_params = None,
                    log_target_reg = False):
    if model_params is None:
        model_params = {}
    
    model = ElasticNet(**model_params)
    pipeline = make_pipeline(StandardScaler(), model)
    if log_target_reg:
        pipeline = _log_target_regressor(pipeline)
    pipeline.fit(X = X_train, y = y_train)

    return pipeline

def _fit_random_forest(X_train, y_train, 
                        obj_func_name = "mse",
                        sample_weight = None, 
                        model_params = None,
                        log_target_reg = False):
    if model_params is None:
        model_params = {}

    # checking if obj_func is correct
    _check_obj_func_name(obj_func_name, "random_forest")
    if "criterion" in model_params.keys():
        assert obj_func_name == model_params["criterion"], "use the obj_func_name argument instead of setting criterion in model_params"
    model_params["criterion"] = obj_func_name

    model = RandomForestRegressor(**model_params)
    if log_target_reg:
        model = _log_target_regressor(model)
    model.fit(X = X_train, y = y_train, sample_weight = sample_weight)

    if log_target_reg:
        if hasattr(model.regressor_, "oob_prediction_"):
            oob_pred = model.regressor_.oob_prediction_
            # must convert back to original units
            oob_pred = model.inverse_func(oob_pred) 
        else:
            oob_pred = None
        model.oob_prediction_ = oob_pred
        model.feature_importances_ = model.regressor_.feature_importances_

    return model


def _fit_lightgbm(X_train, y_train, 
                    obj_func_name = "mse",
                    sample_weight = None, 
                    model_params = None,
                    log_target_reg = False,
                    use_sklearn_api = USE_SKLEARN_API):

    if model_params is None:
        model_params = {}

    # checking if obj_func is correct
    _check_obj_func_name(obj_func_name, "lightgbm")
    if "objective" in model_params.keys():
        assert obj_func_name == model_params["objective"], "use the obj_func_name argument instead of setting objective in model_params"
    model_params["objective"] = obj_func_name

    if use_sklearn_api:
        model = lgb.LGBMRegressor(**model_params)
        if log_target_reg:
            model = _log_target_regressor(model)
        model.fit(X = X_train, y = y_train, sample_weight = sample_weight)
    else:
        if log_target_reg:
            raise NotImplementedError("No implementation of log_target_reg in non-sklearn lightgbm API")
        d_train = lgb.Dataset(data = X_train, label = y_train, weight = sample_weight)
        model = lgb.train(params = model_params, 
                            train_set = d_train)
        model.feature_importances_ = model.feature_importance()

    return model

def _fit_xgboost(X_train, y_train, 
                    obj_func_name = "mse",
                    sample_weight = None, 
                    model_params = None,
                    log_target_reg = False,
                    use_sklearn_api = USE_SKLEARN_API):
                    
    if model_params is None:
        model_params = {}

    # checking if obj_func is correct
    _check_obj_func_name(obj_func_name, "xgboost")
    if "objective" in model_params.keys():
        if obj_func_name != model_params["objective"]:
            assert (obj_func_name == "mse") and (model_params["objective"] == "reg:linear"), "use the obj_func_name argument instead of setting objective in model_params"
    if obj_func_name == "mae":
        # NOTE: this probably doesn't work yet
        obj = _huber_approx_obj
    elif obj_func_name == "mse":
        model_params["objective"] = "reg:linear"
        obj = None
    else:
        model_params["objective"] = obj_func_name
        obj = None

    if use_sklearn_api:
        model_params["n_estimators"] = model_params.get("num_boost_round", 200)
        if obj is not None:
            # override objective in model_params
            model_params["objective"] = obj
        model = xgb.XGBRegressor(importance_type = "gain", **model_params)
        if log_target_reg:
            model = _log_target_regressor(model)
        model.fit(X = X_train, y = y_train, sample_weight = sample_weight)
    else:
        if log_target_reg:
            raise NotImplementedError("No implementation of log_target_reg in non-sklearn xgboost API")
        d_train = xgb.DMatrix(data = X_train, 
                                label = y_train, 
                                weight = sample_weight)
        model = xgb.train(params = model_params, 
                            dtrain = d_train, 
                            obj=obj, 
                            num_boost_round = model_params.get("num_boost_round", 200))
        model.feature_importances_ = model.get_score(importance_type = "gain")


    return model

def _xgb_predict(xgboost_model, X_pred, use_sklearn_api = USE_SKLEARN_API):
    if use_sklearn_api:
        y_pred = xgboost_model.predict(X_pred)
    else:
        X_pred = xgb.DMatrix(X_pred)
        y_pred = xgboost_model.predict(X_pred)

    return y_pred


# clarify what format train_valid_folds should be
def _cv_elastic_net(X_train, y_train, train_valid_folds, 
                    eval_func_names = None, 
                    model_params = None,
                    log_target_reg = False):
    
    if model_params is None:
        model_params = {}
    if eval_func_names is None:
        eval_func_dict = {"r_squared":eval_func_all_dict["sklearn_scorer"]["r_squared"]}
    else:
        assert isinstance(eval_func_names, list), "eval_func_names should be a list"
        eval_func_dict = {eval_func_name:eval_func_all_dict["sklearn_scorer"][eval_func_name] for eval_func_name in eval_func_names}
    
    model = ElasticNet(**model_params)
    pipeline = make_pipeline(StandardScaler(), model)
    if log_target_reg:
        pipeline = _log_target_regressor(pipeline)

    cv_scores =  cross_validate(
                        estimator = pipeline,
                        X = X_train,
                        y = y_train,
                        scoring = eval_func_dict,
                        cv = train_valid_folds,
                        n_jobs = -1,
                        return_train_score = False
                        )

    output = _cv_score_summary(cv_scores)

    return output

def _cv_random_forest(X_train, y_train, train_valid_folds,
                        obj_func_name = "mse",
                        eval_func_names = None,
                        model_params = None, 
                        sample_weight = None,
                        log_target_reg = False,
                        oob_pred = None):
    if model_params is None:
        model_params = {}
    if eval_func_names is None:
        eval_func_dict = {"r_squared":eval_func_all_dict["sklearn_scorer"]["r_squared"]}
    else:
        assert isinstance(eval_func_names, list), "eval_func_names should be a list"
        eval_func_dict = {eval_func_name:eval_func_all_dict["sklearn_scorer"][eval_func_name] for eval_func_name in eval_func_names}
    
    # checking if obj_func is correct
    _check_obj_func_name(obj_func_name, "random_forest")
    if "criterion" in model_params.keys():
        assert obj_func_name == model_params["criterion"], "Found different objective function in model_params. Don't set objective function in model_params, set using obj_func_name in RegressionModel constructor instead"
    model_params["criterion"] = obj_func_name

    model = RandomForestRegressor(**model_params)
    if log_target_reg:
        model = _log_target_regressor(model)

    cv_scores = cross_validate(
                    estimator = model,
                    X = X_train,
                    y = y_train,
                    scoring = eval_func_dict,
                    cv = train_valid_folds,
                    fit_params = {"sample_weight":sample_weight},
                    n_jobs = -1,
                    return_train_score = False
                )

    output = _cv_score_summary(cv_scores)

    if oob_pred is not None:
        metric_func_dict = {eval_func_name:eval_func_all_dict["sklearn_metric"][eval_func_name] for eval_func_name in eval_func_names}
        for eval_func_name, eval_func in metric_func_dict.items():
            oob_score = eval_func(
                            y_true = y_train, 
                            y_pred = oob_pred
                        )
            output["oob-{}".format(eval_func_name)] = oob_score

    return output


# reference for lightgbm cross validation
# https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py
def _cv_lightgbm(X_train, y_train, train_valid_folds, 
                    obj_func_name = "mse",
                    eval_func_names = None,
                    model_params = None,
                    sample_weight = None,
                    log_target_reg = False,
                    use_sklearn_api = USE_SKLEARN_API):

    if model_params is None:
        model_params = {}

    if use_sklearn_api:
        if eval_func_names is None:
            eval_func_dict = {"r_squared":eval_func_all_dict["sklearn_scorer"]["r_squared"]}
        else:
            assert isinstance(eval_func_names, list), "eval_func_names should be a list"
            eval_func_dict = {eval_func_name:eval_func_all_dict["sklearn_scorer"][eval_func_name] for eval_func_name in eval_func_names}
    else:
        if eval_func_names is None:
            eval_func_list = [eval_func_all_dict["lightgbm"]["r_squared"]]
        else:
            assert isinstance(eval_func_names, list), "eval_func_names should be a list"
            eval_func_list = [eval_func_all_dict["lightgbm"][eval_func_name] for eval_func_name in eval_func_names]

    # checking if obj_func is correct
    _check_obj_func_name(obj_func_name, "lightgbm")
    if "objective" in model_params.keys():
        assert obj_func_name == model_params["objective"], "Found different objective function in model_params. Don't set objective function in model_params, set using obj_func_name in RegressionModel constructor instead"
    model_params["objective"] = obj_func_name

    if use_sklearn_api:
        model = lgb.LGBMRegressor(**model_params)
        if log_target_reg:
            model = _log_target_regressor(model)
        
        cv_scores =  cross_validate(
                        estimator = model,
                        X = X_train,
                        y = y_train,
                        scoring = eval_func_dict,
                        cv = train_valid_folds,
                        n_jobs = -1,
                        return_train_score = False,
                        fit_params = {"early_stopping_rounds":100}
                        )

        output = _cv_score_summary(cv_scores)
    else:
        if log_target_reg:
            raise NotImplementedError("No implementation of log_target_reg in non-sklearn lightgbm API")
        d_train = lgb.Dataset(data = X_train, 
                                label = y_train, 
                                weight = sample_weight)
        cv_scores = lgb.cv(
                            params = model_params, 
                            train_set = d_train,
                            num_boost_round = 10000, 
                            folds = train_valid_folds, 
                            early_stopping_rounds = 100,
                            feval = lambda preds, train_data: [eval_func(preds, train_data) for eval_func in eval_func_list],
                            # to ignore the default metric of evaluation function used by objective function
                            metrics = "None"
                        )

        # has mean and std across folds for each boost round
        # there's a post-step of identifying the best number of boosting rounds
        output = {"cv-{}".format(key):val for key,val in cv_scores.items()}
        output = {key.replace("stdv", "std"):val for key,val in output.items()}

    return output

# reference for xgboost
# https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_objective.py
def _cv_xgboost(X_train, y_train, train_valid_folds, 
                obj_func_name = "mse",
                eval_func_names = None,
                model_params= None,
                sample_weight = None,
                log_target_reg = False,
                use_sklearn_api = USE_SKLEARN_API):

    if model_params is None:
        model_params = {}

    if use_sklearn_api:
        if eval_func_names is None:
            eval_func_dict = {"r_squared":eval_func_all_dict["sklearn_scorer"]["r_squared"]}
        else:
            assert isinstance(eval_func_names, list), "eval_func_names should be a list"
            eval_func_dict = {eval_func_name:eval_func_all_dict["sklearn_scorer"][eval_func_name] for eval_func_name in eval_func_names}
    else:
        if eval_func_names is None:
            eval_func = eval_func_all_dict["xgboost"]["r_squared"]
        else:
            assert isinstance(eval_func_names, list), "eval_func_names should be a list"
            eval_func_list = [eval_func_all_dict["xgboost"][eval_func_name] for eval_func_name in eval_func_names]
            eval_func = partial(xgb_eval_func_consolidator, xgb_eval_funcs = eval_func_list)

    # checking if obj_func is correct
    _check_obj_func_name(obj_func_name, "xgboost")
    if "objective" in model_params.keys():
        if obj_func_name != model_params["objective"]:
            assert (obj_func_name == "mse") and (model_params["objective"] == "reg:linear"),"Found different objective function in model_params. Don't set objective function in model_params, set using obj_func_name in RegressionModel constructor instead"
    if obj_func_name == "mae":
        # NOTE: this probably doesn't work yet
        obj = _huber_approx_obj
    elif obj_func_name == "mse":
        model_params["objective"] = "reg:linear"
        obj = None
    else:
        model_params["objective"] = obj_func_name
        obj = None

    if use_sklearn_api:
        model_params["n_estimators"] = model_params.get("num_boost_round", 100000)
        if obj is not None:
            # override objective in model_params
            model_params["objective"] = obj
        model = xgb.XGBRegressor(importance_type = "gain", **model_params)
        if log_target_reg:
            model = _log_target_regressor(model)

        # references on using early stopping with sklearn API
        # https://stackoverflow.com/questions/43866284/grid-search-and-early-stopping-using-cross-validation-with-xgboost-in-scikit-lea?noredirect=1&lq=1
        # https://stackoverflow.com/questions/42993550/gridsearchcv-xgboost-early-stopping/43014158#43014158
        # https://stackoverflow.com/questions/48653038/cross-val-score-for-xgboost-with-early-stopping-rounds-returns-indexerror
        # https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction/discussion/15235

        cv_scores =  cross_validate(
                        estimator = model,
                        X = X_train,
                        y = y_train,
                        scoring = eval_func_dict,
                        cv = train_valid_folds,
                        n_jobs = -1,
                        return_train_score = False,
                        fit_params = {"early_stopping_rounds":100}
                        )

        output = _cv_score_summary(cv_scores)
    else:
        if log_target_reg:
            raise NotImplementedError("No implementation of log_target_reg in non-sklearn xgboost API")
        d_train = xgb.DMatrix(
                    data = X_train, 
                    label = y_train, 
                    weight = sample_weight
                    )
        cv_scores = xgb.cv(
                        params = model_params, 
                        dtrain = d_train, 
                        num_boost_round = 100000, 
                        folds = train_valid_folds, 
                        early_stopping_rounds = 100,
                        obj = obj, 
                        feval = eval_func
                    )

        output = cv_scores.to_dict(orient = "list")
        output = {key.replace("test", "cv"):value for key,value in output.items() if "test-" in key}


    return output