from preprocessor import FeaturePreprocessor
from regression_model import RegressionModel
from hyperparameter_optimizer import HyperParameterOptimizer
from diagnostics import HPODiagnoser, ModelDiagnoser
from interpret import ModelInterpreter

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold


def preprocessor_test():
    pass

def regression_model_test():
    X, y = datasets.load_boston(return_X_y = True)
    X = pd.DataFrame(X)

    model_name = "random_forest"
    random_seed = 1001
    obj_func_name = "mse"
    eval_func_names = ["r_squared", "rmse"]
    n_estimators = 400

    model = RegressionModel(X_train = X, y_train = y, model_type = model_name, 
                        obj_func_name = obj_func_name, random_seed = random_seed)

    if model_name == "random_forest":
        model_params = {"n_estimators":n_estimators}
    else:
        model_params = {}

    model.fit(model_params = model_params)

    training_set_preds = model.predict(X)
    print(training_set_preds)

    cv_metrics = model.cross_validate(train_valid_folds = 10,
                                    eval_func_names = eval_func_names,
                                    model_params = model_params)
    print(cv_metrics)

def hyperparameter_optimizer_test():
    X, y = datasets.load_boston(return_X_y = True)
    X = pd.DataFrame(X)

    model_name = "random_forest"
    random_seed = 1001
    obj_func_name = "mse"
    n_estimators = 400
    total_n_iterations = 50

    base_model = RegressionModel(X_train = X, y_train = y, model_type = model_name, 
                        obj_func_name = obj_func_name, random_seed = random_seed)

    hpo = HyperParameterOptimizer(verbosity = 1)
    if model_name == "random_forest":
        override_params = {"n_estimators":n_estimators}
    else:
        override_params = {}
    hpo.tune_and_fit(model = base_model,
                    total_n_iterations = total_n_iterations,
                    train_valid_folds = 10,
                    override_params = override_params,
                    use_model_copy = True)
    tuned_model = hpo.model
    return tuned_model

def model_diagnoser_test():
    X, y = datasets.load_boston(return_X_y = True)
    X = pd.DataFrame(X)

    model_name = "random_forest"
    random_seed = 1001
    obj_func_name = "mse"
    eval_func_names = ["r_squared", "rmse"]
    n_estimators = 400

    X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size = 0.2, random_state = random_seed)

    model = RegressionModel(X_train = X_train, y_train = y_train, model_type = model_name, 
                        obj_func_name = obj_func_name, random_seed = random_seed)

    if model_name == "random_forest":
        model_params = {"n_estimators":n_estimators}
    else:
        model_params = {}

    model.fit(model_params = model_params)

    model_diagnoser = ModelDiagnoser(model, 
                                    train_valid_folds = 10,
                                    eval_func_names = eval_func_names,
                                    X_test = X_test,
                                    y_test = y_test)
    model_diagnoser.show_all_diagnostics()

def hpo_diagnoser_test():
    X, y = datasets.load_boston(return_X_y = True)
    X = pd.DataFrame(X)

    model_name = "random_forest"
    random_seed = 1001
    obj_func_name = "mse"
    n_estimators = 400
    total_n_iterations = 50

    base_model = RegressionModel(X_train = X, y_train = y, model_type = model_name, 
                        obj_func_name = obj_func_name, random_seed = random_seed)

    hpo = HyperParameterOptimizer(verbosity = 1)
    if model_name == "random_forest":
        override_params = {"n_estimators":n_estimators}
    else:
        override_params = {}
    hpo.tune_and_fit(model = base_model,
                    total_n_iterations = total_n_iterations,
                    train_valid_folds = 10,
                    override_params = override_params,
                    use_model_copy = True)

    hpo_diagnoser = HPODiagnoser(hpo)
    figures = hpo_diagnoser.show_all_diagnostics()
    
    return figures

def run_all_tests():
    regression_model_test()
    print("regression_model_test successful!")

    hyperparameter_optimizer_test()
    print("hyperparameter_optimizer_test successful!")

    model_diagnoser_test()
    print("model_diagnoser_test successful!")

    _ = hpo_diagnoser_test()
    plt.show()
    print("hpo_diagnoser_test successful!")

    print("all tests successful!")
