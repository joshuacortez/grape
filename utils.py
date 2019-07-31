import ast
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer

# todo: get ndcg

# taken from https://stackoverflow.com/questions/45006341/xgboost-how-to-use-mae-as-objective-function
def _huber_approx_obj(preds, dtrain):
    d = preds - dtrain.get_label() #remove .get_labels() for sklearn
    h = 1  #h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = (1 / scale) / scale_sqrt
    return grad, hess


def str_to_dict(x):
    if type(x) == str:
        x = ast.literal_eval(x)
    return x.copy()

def linear_penalty_type(l1_ratio):
    if l1_ratio == 1:
        penalty_type = "lasso"
    elif l1_ratio == 0:
        penalty_type = "ridge"
    else:
        penalty_type = "elastic_net"
    return penalty_type

def get_sample_weight(samples, metric):
    '''
    Positional Arguements:
        samples: Numpy Array. Required.
        metric: String. 'l1' or 'l2'. Required.
    '''
    # TODO: Raise an appropriate error instead of AssertionError
    assert metric in ["l1", "l2"]
        
    if metric == "l1":
        # weight the observations so the training can implicitly optimize for Mean Absolute Percentage Error (MAPE) 
        # instead of Mean Absolute Error (MAE)
        sample_weight = samples.apply(lambda y: 1/y)

    elif metric == "l2":
        # weight the observations so the training can implicitly optimize for Mean Squared Percentage Error (MSPE) 
        # instead of Mean Squared Error (MSE)
        sample_weight = samples.apply(lambda y: 1/(y**2))
        
    # normalize the weights to sum up to 1.0
    sample_weight = sample_weight/sample_weight.sum()
    
    # turn into numpy array
    sample_weight = sample_weight.get_values()
    
    return sample_weight

def get_elastic_net_l1_ratio(model_params):
    penalty_type = model_params["l1_ratio"]["penalty_type"]
    if penalty_type == "lasso":
        l1_ratio = 1
    elif penalty_type == "ridge":
        l1_ratio = 0
    else:
        l1_ratio = model_params["l1_ratio"]["l1_ratio"]

    return l1_ratio

r2_scorer = make_scorer(r2_score)
mse_scorer = make_scorer(mean_squared_error)
mae_scorer = make_scorer(mean_absolute_error)

def r2_eval_func_lgb(preds, train_data):
    y_true = train_data.get_label()
    r2 = r2_score(y_true = y_true, y_pred = preds)
    eval_name, eval_result, is_higher_better = "r_squared", r2, True
    return eval_name, eval_result, is_higher_better

def mse_eval_func_lgb(preds, train_data):
    y_true = train_data.get_label()
    mse = mean_squared_error(y_true = y_true, y_pred = preds)
    eval_name, eval_result, is_higher_better = "mse", mse, False
    return eval_name, eval_result, is_higher_better
    
def mae_eval_func_lgb(preds, train_data):
    y_true = train_data.get_label()
    mae = mean_absolute_error(y_true = y_true, y_pred = preds)
    eval_name, eval_result, is_higher_better = "mae", mae, False
    return eval_name, eval_result, is_higher_better

def r2_eval_func_xgb(preds, train_data):
    y_true = train_data.get_label()
    r2 = r2_score(y_true = y_true, y_pred = preds)
    eval_name, eval_result = "r_squared", r2
    return eval_name, eval_result

def mse_eval_func_xgb(preds, train_data):
    y_true = train_data.get_label()
    mse = mean_squared_error(y_true = y_true, y_pred = preds)
    eval_name, eval_result = "mse", mse
    return eval_name, eval_result
    
def mae_eval_func_xgb(preds, train_data):
    y_true = train_data.get_label()
    mae = mean_absolute_error(y_true = y_true, y_pred = preds)
    eval_name, eval_result = "mae", mae
    return eval_name, eval_result

# workaround for multiple scoring functions on cv 
# https://github.com/dmlc/xgboost/issues/1125
def xgb_eval_func_consolidator(preds, train_data, xgb_eval_funcs):
    results = [xgb_eval_func(preds, train_data) for xgb_eval_func in xgb_eval_funcs]
    return results

eval_func_all_dict = {
    "sklearn_metric":{"r_squared":r2_score, "mse":mean_squared_error, "mae":mean_absolute_error},
    "sklearn_scorer":{"r_squared":r2_scorer, "mse":mse_scorer,"mae":mae_scorer},
    "lightgbm":{"r_squared":r2_eval_func_lgb, "mse":mse_eval_func_lgb, "mae":mae_eval_func_lgb},
    "xgboost":{"r_squared":r2_eval_func_xgb, "mse":mse_eval_func_xgb, "mae":mae_eval_func_xgb},
}

eval_func_is_higher_better = {
    "r_squared":True,
    "mse":False,
    "mae":False
}
