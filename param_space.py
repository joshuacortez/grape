import numpy as np
import hyperopt

parameter_space = {
    "elastic_net":{
        # adjusted the end of alpha to be 15 from 20
        # needs a better way of specifying alpha, similar to how its done in elasticnetcv
        "alpha":hyperopt.hp.loguniform("alpha", np.log(0.01), np.log(15)),
        "l1_ratio":hyperopt.hp.choice(
            "penalty_type",
            [
                {"penalty_type":"lasso"},
                {"penalty_type":"ridge"},
                {"penalty_type":"elastic_net", "l1_ratio":hyperopt.hp.uniform("l1_ratio", 0, 1)}
            ]
        )
    },
    "random_forest":{
        # allow up to 20 (based on auto-sklearn search space) to allow for regularization
        'min_samples_split':hyperopt.hp.quniform('min_samples_split', 2, 20, 1),
        # allow up to 20 (based on auto-sklearn search space) to allow for regularization
        'min_samples_leaf':hyperopt.hp.quniform('min_samples_leaf', 1, 20, 1),
        # allow from 0% to 100% max_features (based on auto-sklearn search space) to allow for regularization
        'max_features':hyperopt.hp.uniform('max_features', 0.0, 1.0) 
    },
    "lightgbm":{
        'min_child_samples': hyperopt.hp.quniform('min_child_samples', 2, 350, 2),
        'reg_lambda': hyperopt.hp.uniform('reg_lambda', 0.0, 1.0),
        'colsample_bytree': hyperopt.hp.uniform('colsample_by_tree', 0.0, 1.0)
    },
    # NOTE: tuning xgboost
    # https://www.kaggle.com/c/santander-customer-satisfaction/discussion/20662
    "xgboost":{
        'min_child_weight': hyperopt.hp.quniform('min_child_weight', 1, 350, 1),
        'reg_lambda': hyperopt.hp.uniform('reg_lambda', 0.0, 1.0),
        #'colsample_bytree': hyperopt.hp.uniform('colsample_by_tree', 0.0, 1.0),
        'gamma': hyperopt.hp.uniform("gamma", 0.0, 2.0),
    }
}

int_params_dict = {
    "elastic_net":[],
    "random_forest":["min_samples_split", "min_samples_leaf", "n_estimators"],
    "lightgbm":['num_leaves', 'subsample_for_bin', 'min_child_samples'],
    "xgboost":["min_child_weight"],
}