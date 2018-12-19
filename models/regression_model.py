import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV

class Model:
    def __init__(self, model_type, random_state, metric, model_params_dict = None):
        assert model_type in ["random_forest","elastic_net", "lightgbm", "lightgbm_special", "xgboost"]
        
        self._model_type = model_type
  
        # this is a dictionary of hyperparameters for the models
        if model_params_dict is None:
            model_params_dict = {}
        self._model_params_dict = model_params_dict
        if model_type == "xgboost":
            self._model_params_dict["seed"] = random_state
        else:
            self._model_params_dict["random_state"] = random_state
        
        if model_type in ["random_forest", "elastic_net", "xgboost"]:
            if metric == "l1":
                self._metric = "mae"
            if metric == "l2":
                self._metric = "mse"
        if model_type == "lightgbm":
            self._metric = "regression_{}".format(metric)
        if model_type == "lightgbm_special":
            self._metric = metric
                
    def fit(self, X_train, y_train, sample_weight = None, train_valid_folds = None):
        if train_valid_folds is not None:
            assert self._model_type == "elastic_net", "Auto CV (non-bayesian opt) only implemented for elastic_net"
        
        if self._model_type == "elastic_net":
            
            # fit model
            # passing a train_valid_folds object leads to grid search cross validation
            if train_valid_folds is not None:
                self._model_params_dict["n_jobs"] = -1
                cv = train_valid_folds.split(X_train)
                #cv = train_valid_folds.make_train_valid_folds(X_train)
                model = ElasticNetCV(**self._model_params_dict, cv = cv)
            else:
                model = ElasticNet(**self._model_params_dict)
            model.fit(X = X_train, y = y_train)
            
            feature_importance_df = pd.DataFrame(data = {"feature_importance":model.coef_}, index = X_train.columns)
            

        if self._model_type == "random_forest":
            self._model_params_dict["n_jobs"] = -1
            self._model_params_dict["criterion"] = self._metric
            model = RandomForestRegressor(**self._model_params_dict)
            
            # fit model
            model.fit(X = X_train, y = y_train, sample_weight = sample_weight)
            
            feature_importance_df = pd.DataFrame(data = {"feature_importance":model.feature_importances_}, 
                                                 index = X_train.columns)
                
        if (self._model_type == "lightgbm") | (self._model_type == "lightgbm_special"):
            self._model_params_dict["n_jobs"] = -1
            # not really a metric (really the objective function but yeah)
            
            if self._model_type == "lightgbm":
                self._model_params_dict["objective"] = self._metric
                fobj = None
                feval = None
            if self._model_type == "lightgbm_special":
                print("Ignoring the set metric {} for lightgbm_special".format(self._metric))
                if sample_weight is not None:
                    print("Disregarding sample_weights (for mape or mspe) for lightgbm_special")
                    sample_weight = None
                self._model_params_dict["metric"] = "None"
                fobj = wiley_fobj
                feval = wiley_feval
            
            d_train = lgb.Dataset(data = X_train, label = y_train, weight = sample_weight)
            
            # fit model
            model = lgb.train(params = self._model_params_dict, train_set = d_train, 
                              fobj = fobj, feval = feval)
            feature_importance_df = pd.DataFrame(data = {"feature_importance":model.feature_importance()}, 
                                     index = X_train.columns)
    
        if self._model_type == "xgboost":
            self._model_params_dict["silent"] = 1
            
            if self._metric == "mae":
                obj = self._huber_approx_obj
                obj = None
            if self._metric == "mse":
                obj = None
                #self._model_params_dict["objective"] = "reg:linear"
            
            if sample_weight is not None:
                print("Sample weight not yet supported with the XGBoost model")
                sample_weight = None
            
            d_train = xgb.DMatrix(data = X_train, label = y_train, weight = sample_weight)
            # fit model
            model = xgb.train(params = self._model_params_dict, dtrain = d_train, 
                              obj=obj, num_boost_round = self._model_params_dict.get("num_boost_round", 200)) 
            feature_importance_df = pd.DataFrame.from_dict({"feature_importance":model.get_score(importance_type = "gain")})
        
        feature_importance_df = feature_importance_df.sort_values(by = "feature_importance", ascending = False)
        self.feature_importance_df = feature_importance_df
        self.model = model
        
    def predict(self, X_test):
        if self._model_type == "xgboost":
            X_test = xgb.DMatrix(X_test)
        
        predictions = self.model.predict(X_test)
        
        return predictions
    
    @staticmethod
    # taken from https://stackoverflow.com/questions/45006341/xgboost-how-to-use-mae-as-objective-function
    def _huber_approx_obj(preds, dtrain):
        d = preds - dtrain.get_label() #remove .get_labels() for sklearn
        h = 1  #h is delta in the graphic
        scale = 1 + (d / h) ** 2
        scale_sqrt = np.sqrt(scale)
        grad = d / scale_sqrt
        hess = (1 / scale) / scale_sqrt
        return grad, hess

def test():
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_boston
    from sklearn.metrics import r2_score
    RANDOM_SEED =  46257
    
    boston = load_boston()
    df = pd.DataFrame(data = boston["data"])
    df.columns = ["var_{}".format(col) for col in df.columns]
    df["target"] = boston["target"]
    train_df, test_df = train_test_split(df, test_size = 0.2, random_state = RANDOM_SEED)
    
    attribute_cols_list = [col for col in df.columns if col != "target"]
    y_cols_list = ["target"]
    USE_WEIGHT_SAMPLES = False
    categorical_vars_list = []
    
    for model_type in ["elastic_net", "random_forest", "xgboost", "lightgbm"]:
        print("fitting baseline {} model".format(model_type))
        model = Model(model_type = model_type, random_state = RANDOM_SEED,
                      metric = "l2")
        model.fit(X_train = train_df.loc[:,attribute_cols_list],
                 y_train = train_df["target"])
    
        predictions = model.predict(X_test = test_df.loc[:,attribute_cols_list])
        r2 = r2_score(test_df["target"], predictions)
        print("R-Squared on test set predictions on boston dataset is {}".format(r2))

if __name__ == "__main__":
    test()