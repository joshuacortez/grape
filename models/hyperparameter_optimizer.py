from param_space import model_params_space_dict

class HyperparameterOptimizer:
    # uses bayesian optimization to optimize hyparmareters for machine learning models
    def __init__(self, model_type, X_train, y_train, y_colname, metric, random_state, train_valid_folds,
                sample_weight = None, output_folder = None):
        assert model_type in ["random_forest", "lightgbm", "elastic_net", "xgboost"]
        assert metric in ["l1", "l2"]
        
        self._model_type = model_type
        self._random_state = random_state
        self._y_colname = y_colname
        self._train_valid_folds = list(train_valid_folds.split(X_train))
        #self._train_valid_folds = train_valid_folds.make_train_valid_folds(X_train)
        self._sample_weight = sample_weight
        self._output_folder = output_folder

        # Keep track of iterations of hyperparameter search
        self.num_iterations = 0
    
        # prepare dataset differently depending on the algorithm used 
        if (self._model_type == "random_forest") | (self._model_type == "elastic_net"):
            if metric == "l1":
                self._metric = "mae"
                self._scoring = "neg_mean_absolute_error"
            if metric == "l2":
                self._metric = "mse"
                self._scoring = "neg_mean_squared_error"
            # transform datasets from pandas dataframe/series to matrix (hence the get_values())
            self._X_train = X_train.get_values()
            self._y_train = y_train.get_values()
            
        if self._model_type == "lightgbm":
            self._metric = "regression_{}".format(metric)
            
            d_train = lgb.Dataset(data = X_train, label = y_train, weight = sample_weight)
            self.d_train = d_train
            
        if self._model_type == "xgboost":
            if sample_weight is not None:
                print("Sample weight not yet supported with the XGBoost model")
                sample_weight = None
            
            d_train = xgb.DMatrix(data = X_train, label = y_train, weight = sample_weight)
            self.d_train = d_train
            if metric == "l1":
                self._metric = "mae"
            if metric == "l2":
                self._metric = "mse"

        
    def _objective_elasticnet(self, model_params):
        """Objective function for Elastic Net Regression Hyperparameter Optimization"""
        
        self.num_iterations += 1
        print("num_iterations {}".format(self.num_iterations))
        
        # ----------- conditional sampling from bayesian domain for l1_ratio
        penalty_type = model_params["l1_ratio"]["penalty_type"]
        if penalty_type == "lasso":
            l1_ratio = 1
        elif penalty_type == "ridge":
            l1_ratio = 0
        else:
            l1_ratio = model_params["l1_ratio"]["l1_ratio"]
        model_params["l1_ratio"] = l1_ratio
        # ----------- conditional sampling from bayesian domain for l1_ratio

        start = timer()
    
        # set up model
        model = ElasticNet(max_iter = 10000, random_state = self._random_state, **model_params)
        
        # make a pipeline that performs z-transformation before fitting the model within the cross validation loop
        # this is to prevent leakage into each validation set in each fold
        # weakness of this is that the z-transformation happens every time
        # https://chrisalbon.com/machine_learning/model_evaluation/cross_validation_pipeline/
        
        pipeline = make_pipeline(StandardScaler(), model)
        cross_val_scores = cross_val_score(estimator = pipeline, X = self._X_train, y = self._y_train,
                                           scoring = self._scoring, cv = self._train_valid_folds,
                                          n_jobs = -1)
        # multiply by negative one because the sklearn scoring metrics used are in negative (i.e. neg_mean_squared_error)
        loss = np.mean(cross_val_scores)*(-1)
        
        run_time = timer() - start   
        
        headers = ['loss', 'params', 'num_iterations', 'train_time']
        eval_vals = [loss, model_params, self.num_iterations, run_time]
        self._dump_params(headers, eval_vals)
        
        # Dictionary with information for evaluation
        return {'loss': loss, 'params': model_params, 'num_iterations': self.num_iterations,
                'train_time': run_time, 'status': hyperopt.STATUS_OK}
        
    def _objective_rf(self, model_params):
        """Objective function for Random Forest Regression Hyperparameter Optimization"""
    
        self.num_iterations += 1
        print("num_iterations {}".format(self.num_iterations))

        start = timer()
        
        model_params["criterion"] = self._metric
        # dont pre-select the number of estimators
        # model_params["n_estimators"] = 400
        # convert to integer the hyparameters that are supposed to be integers
        for param in ["min_samples_split", "min_samples_leaf", "n_estimators"]:
            if param in model_params.keys():
                model_params[param] = int(model_params[param])

        # set up model
        model = RandomForestRegressor(n_jobs = -1, random_state = self._random_state, **model_params)
        
        oob_score = model_params.get("oob_score", False)
        if oob_score:
            # if using out-of-bag error instead of cross validation error
            # https://stats.stackexchange.com/questions/207815/out-of-bag-error-makes-cv-unnecessary-in-random-forests
            model.fit(X = self._X_train, y = self._y_train, 
                            sample_weight = self._sample_weight)

            if self._metric == "mae":
                loss = mean_absolute_error(y_true = self._y_train, y_pred = model.oob_prediction_)
            if self._metric == "mse":
                loss = mean_squared_error(y_true = self._y_train, y_pred = model.model.oob_prediction_)
        else:
            # if not using oob score
            # use if cross validation is not simply taking a random subset
            # see http://www.fast.ai/2017/11/13/validation-sets/
            cross_val_scores = cross_val_score(estimator = model, X = self._X_train, y = self._y_train,
                                               scoring = self._scoring, cv = self._train_valid_folds,
                                              fit_params = {"sample_weight":self._sample_weight},
                                              n_jobs = -1)
            # multiply by negative one because the sklearn scoring metrics used are in negative (i.e. neg_mean_squared_error)
            loss = np.mean(cross_val_scores)*(-1)
        
            
        run_time = timer() - start   
        
        headers = ['loss', 'params', 'num_iterations', 'train_time']
        eval_vals = [loss, model_params, self.num_iterations, run_time]
        self._dump_params(headers, eval_vals)

        # Dictionary with information for evaluation
        return {'loss': loss, 'params': model_params, 'num_iterations': self.num_iterations,
                'train_time': run_time, 'status': hyperopt.STATUS_OK}
    
    def _objective_lgb(self, model_params):
        """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
    
        self.num_iterations += 1
        print("num_iterations {}".format(self.num_iterations))

        if self._model_type == "lightgbm":
            model_params["objective"] = self._metric
            ## to change: maybe fobj and feval should be better explicitly stated instead of implied by the objective parameter
            fobj = None
            feval = None

        
        # ----------- conditional sampling from bayesian domain for the goss boosting type
        if "boosting_type" in model_params.keys():
            # Retrieve the subsample if present otherwise set to 1.0
            subsample = model_params['boosting_type'].get('subsample', 1.0)

            # Extract the boosting type
            model_params['boosting_type'] = model_params['boosting_type']['boosting_type']
            model_params['subsample'] = subsample
        # ----------- conditional sampling from bayesian domain for the goss boosting type

        # Make sure parameters that need to be integers are integers
        for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
            if parameter_name in model_params.keys():
                model_params[parameter_name] = int(model_params[parameter_name])

        start = timer()
        
        # Perform cross validation
        cv_results = lgb.cv(params = model_params, train_set = self.d_train, num_boost_round = 10000, 
                            folds = self._train_valid_folds, early_stopping_rounds = 100, 
                            seed = self._random_state, fobj = fobj, feval = feval)

        run_time = timer() - start

        # Extract the best loss
        metric_mean_name = [key for key in cv_results.keys() if "-mean" in key][0]
        metric_std_name = [key for key in cv_results.keys() if "-std" in key][0]

        loss = np.min(cv_results[metric_mean_name])
        std_loss = cv_results[metric_std_name][np.argmin(cv_results[metric_mean_name])]

        # Boosting rounds that returned the best cv score
        num_boost_round = int(np.argmin(cv_results[metric_mean_name]) + 1)

        headers = ['loss', 'std_loss', 'params', 'num_iterations', 'num_boost_round', 'train_time']
        eval_vals = [loss, std_loss, model_params, self.num_iterations, num_boost_round, run_time]
        self._dump_params(headers, eval_vals)
    
        # Dictionary with information for evaluation
        return {'loss': loss, "std_loss":std_loss, 'params': model_params, 'num_iterations': self.num_iterations,
                'estimators': num_boost_round, 'train_time': run_time, 'status': hyperopt.STATUS_OK}
    
    def _objective_xgb(self, model_params):
        """Objective function for Extreme Gradient Boosting Machine Hyperparameter Optimization"""
        
        self.num_iterations += 1
        print("num_iterations {}".format(self.num_iterations))
        
        if self._metric == "mae":
            obj = self._huber_approx_obj
            metric = "mae"
        elif self._metric == "mse":
            model_params["objective"] = "reg:linear"
            obj = None
            metric = "rmse"
        else:
            print("wrong metric")
                
        # Make sure parameters that need to be integers are integers
        for parameter_name in ['min_child_weight']:
            if parameter_name in model_params.keys():
                model_params[parameter_name] = int(model_params[parameter_name])
                
        model_params["silent"] = 1
            
        start = timer()

        # Perform cross validation
        cv_results = xgb.cv(params = model_params, dtrain = self.d_train, num_boost_round = 100000, 
                            folds = self._train_valid_folds, early_stopping_rounds = 100, 
                            seed = self._random_state, obj = obj, metrics = metric)

        run_time = timer() - start
        
        # Extract the best loss
        metric_mean_name = "test-{}-mean".format(metric)
        metric_std_name = "test-{}-std".format(metric)

        best_idx = cv_results[metric_mean_name].idxmin()
        loss = cv_results.loc[best_idx,metric_mean_name]
        std_loss = cv_results.loc[best_idx,metric_std_name]

        # Boosting rounds that returned the best cv score
        num_boost_round = best_idx + 1
        
        headers = ['loss', 'std_loss', 'params', 'num_iterations', 'num_boost_round', 'train_time']
        eval_vals = [loss, std_loss, model_params, self.num_iterations, num_boost_round, run_time]
        self._dump_params(headers, eval_vals)
    
        # Dictionary with information for evaluation
        return {'loss': loss, "std_loss":std_loss, 'params': model_params, 'num_iterations': self.num_iterations,
                'num_boost_round': num_boost_round, 'train_time': run_time, 'status': hyperopt.STATUS_OK}
    
    # using huber loss in place of having no built-in mae loss in xgboost
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
    
    def _dump_params(self, headers, eval_vals):
        if self.trial_summary_outfile is not None:
            if self.num_iterations == 1:
                with open(self.trial_summary_outfile, 'w') as file:
                    writer = csv.writer(file)
                    # Write the headers to the file
                    writer.writerow(headers)

            # Write to the csv file ('a' means append)
            with open(self.trial_summary_outfile, 'a') as file:
                writer = csv.writer(file)
                writer.writerow(eval_vals)
            
        eval_dict = {key:val for key,val in zip(headers,eval_vals)}
        self.trial_runs.append(eval_dict)
        
    def run_hyperparameter_tuning(self, parameter_space_dict, max_evals):
        # File to save first results
        timenow = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        if self._output_folder is not None:
            self.trial_summary_outfile = '{}/trial-summary_{}_{}_{}.csv'.format(self._output_folder, self._model_type, 
                                                                           self._y_colname, timenow)
        else:
            self.trial_summary_outfile = None
        
        # Keep track of results
        # The Trials object will hold everything returned from the objective function in the .results attribute. 
        # It also holds other information from the search, but we return everything we need from the objective
        bayes_trials = hyperopt.Trials()

        # filename trial results (we can load this later as a checkpoint for further hyperparameter tuning)
        bayes_trials_outfile = "{}/bayes-trials_{}_{}_{}.json".format(self._output_folder, self._model_type, 
                                                                      self._y_colname, timenow)
        self.bayes_trials_outfile = bayes_trials_outfile
        
        # create a partial function with the other arguments already passed in
        if (self._model_type == "lightgbm") | (self._model_type == "lightgbm_special"):
            objective = self._objective_lgb
        if self._model_type == "random_forest":
            objective = self._objective_rf
        if self._model_type == "elastic_net":
            objective = self._objective_elasticnet
        if self._model_type == "xgboost":
            objective = self._objective_xgb
            
        # list that contains all the results of bayesian optimization
        self.trial_runs = []
            
        # Run optimization
        best = hyperopt.fmin(fn = objective, space = parameter_space_dict, algo = hyperopt.tpe.suggest, 
                    max_evals = max_evals, trials = bayes_trials, rstate = np.random.RandomState(self._random_state))
        
        # save the trial results (don't do this yet, will fix after worked on checkpointing)
        #with open(bayes_trials_outfile, 'w') as file:
        #    json.dump(bayes_trials.results, file)
            
        print("{} evals of Bayesian optimization done".format(max_evals))
        if self.trial_summary_outfile is not None:
            print("Summary file for every num_iterations can be found in {}".format(self.trial_summary_outfile))
        # don't do this yet, will fix after worked on checkpointing
        #print("Trial Results file (for checkpointing) can be find in {}".format(bayes_trials_outfile))
        
        return self.trial_runs

        
def test():
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_boston
    RANDOM_SEED =  46257

    boston = load_boston()
    df = pd.DataFrame(data = boston["data"])
    df.columns = ["var_{}".format(col) for col in df.columns]
    df["target"] = boston["target"]
    train_df, test_df = train_test_split(df, test_size = 0.2, random_state = RANDOM_SEED)
    train_valid_folds = KFold(n_splits = 5, shuffle = True, random_state = RANDOM_SEED)

    attribute_cols_list = [col for col in df.columns if col != "target"]
    y_cols_list = ["target"]
    USE_WEIGHT_SAMPLES = False
    categorical_vars_list = []

    for model_type in ["elastic_net", "random_forest", "lightgbm", "xgboost"]:
        hpo = HyperparameterOptimizer(model_type = model_type, 
                                      X_train = train_df.loc[:,attribute_cols_list],
                                     y_train = train_df.loc[:,["target"]],
                                     y_colname = "target",
                                     metric = "l2",
                                     random_state = RANDOM_SEED,
                                      train_valid_folds = train_valid_folds,
                                      output_folder = None
                                     )
        print(model_type)
        results = hpo.run_hyperparameter_tuning(parameter_space_dict = model_params_space_dict[model_type], max_evals = 10)
        results = pd.DataFrame(results)
        print(results.head())
    
if __name__ == "__main__":
    test()
