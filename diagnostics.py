import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hyperopt
import scipy.stats

from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import KFold

from param_space import parameter_space
from utils import str_to_dict, linear_penalty_type

RANDOM_SEED_ARGNAMES = ["random_state", "seed"]

class ModelDiagnoser:
    def __init__(self, model, 
                     train_valid_folds = None,
                     X_test = None,
                     y_test = None):

        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.train_valid_folds = train_valid_folds
        assert hasattr(model, "X_train") and hasattr(model, "y_train"), "Need training set bound to model"

        self.get_model_diagnostics()

    def get_model_diagnostics(self):

        assert (self.X_test is None) == (self.y_test is None), "X_test and y_test arguments have to be both supplied or neither"

        self.model_diagnostics = {}

        training_preds = self.model.predict(self.model.X_train)

        self.model_diagnostics["training-r_squared"] = r2_score(y_true = self.model.y_train, 
                                                                y_pred = training_preds)
        if hasattr(self.model.model, "oob_score_"):
            self.model_diagnostics["oob-r_squared"] = self.model.model.oob_score_



        if self.train_valid_folds is not None:
            model_params = {key:val for key,val in self.model.model_params.items() if key not in RANDOM_SEED_ARGNAMES}
            cv_scores = self.model.cross_validate(train_valid_folds = self.train_valid_folds,
                                                  eval_func_names = "r_squared",
                                                 model_params = model_params)
            r_squared_mean = cv_scores["cv-r_squared-mean"]
            r_squared_std = cv_scores["cv-r_squared-std"]
            if self.model.model_type in ["lightgbm", "xgboost"]:
                best_idx = np.argmax(r_squared_mean)
                r_squared_mean = r_squared_mean[best_idx]
                r_squared_std = r_squared_std[best_idx]
            
            self.model_diagnostics["cv-r_squared-mean"] = r_squared_mean
            self.model_diagnostics["cv-r_squared-std"] = r_squared_std

        if (self.X_test is not None) & (self.y_test is not None):
            test_preds = self.model.predict(self.X_test)
            self.model_diagnostics["test-r_squared"] = r2_score(y_true = self.y_test, y_pred = test_preds)
    
    # def plot_actual_vs_predicted(self, X, y):
    #     preds = self.model.model.predict(X)

    #     return preds

class HPODiagnoser:

    def __init__(self, hpo):
        self.hpo = hpo

        # find the index of minimum loss
        min_loss_idx = self.hpo.hyperopt_summary["loss"].idxmin()
        # get the parameter dictionary from the iteration with the lowest loss
        best_params_dict =  str_to_dict(self.hpo.hyperopt_summary.loc[min_loss_idx,"params"])
        
        if self.hpo.model.model_type == "lightgbm":
            # include number of boosting rounds from hyperparameter optimization
            best_params_dict["num_iterations"] = self.hpo.hyperopt_summary.loc[min_loss_idx,"estimators"]
        if self.hpo.model.model_type == "xgboost":
            best_params_dict["num_boost_round"] = self.hpo.hyperopt_summary.loc[min_loss_idx,"num_boost_round"]
        self.best_params_dict_ = best_params_dict
        self.best_loss_ = self.hpo.hyperopt_summary.loc[min_loss_idx,"loss"]

    def get_hyperparam_summary(self):
        # Create a new empty dataframe for storing parameters
        columns = list(str_to_dict(
            self.hpo.hyperopt_summary.loc[0, 'params']
        ).keys())
        
        bayes_params_df = pd.DataFrame(
            columns = columns,
            index = list(range(len(self.hpo.hyperopt_summary)))
        )
        # Add the results with each parameter a different column
        for i, params in enumerate(self.hpo.hyperopt_summary['params']):
            bayes_params_df.loc[i, :] = list(str_to_dict(params).values())

        for colname in self.hpo.hyperopt_summary.columns.tolist():
            if colname != "params":
                bayes_params_df[colname] = self.hpo.hyperopt_summary[colname]

        return bayes_params_df

    def plot_bayes_hyperparam_density(self, parameter_name, n_samples = 1000):
        bayes_params_df = self.get_hyperparam_summary()
        parameter_space_dict = parameter_space[self.hpo.model.model_type]
        if parameter_name in parameter_space_dict.keys():
        
            fig, ax = plt.subplots()
            # plot the density from bayes optimization
            sns.kdeplot(
                bayes_params_df[parameter_name], 
                label = "Bayes Opt Distribution", 
                ax = ax, 
                color = "blue"
            )
            ax.axvline(
                bayes_params_df[parameter_name].median(), 
                linestyle = "--", 
                color = "blue", 
                alpha = 0.8
            )

            # plot the density from the sampling distribution
            parameter_samples = [
                hyperopt.pyll.stochastic.sample(parameter_space_dict[parameter_name]) for i in range(n_samples)
            ]
            sns.kdeplot(
                parameter_samples, 
                label = "Sampling Distribution", 
                ax = ax, 
                color = "orange"
            )
            ax.axvline(
                np.median(parameter_samples), 
                linestyle = "--", 
                color = "orange", 
                alpha = 0.8
            )

            ax.axvline(
                self.best_params_dict_[parameter_name], 
                label = "Best {}".format(parameter_name), 
                linestyle = "--", 
                color = "green", 
                alpha = 0.8
            )

            ax.set_ylabel("Density")
            ax.set_xlabel(parameter_name)
            ax.legend()
            return fig, ax
        else:
            print("{} not found among hyperparameters in parameter_space_dict".format(parameter_name))

    def plot_param_over_iterations(self, parameter_name):
        bayes_params_df = self.get_hyperparam_summary()
        best_iteration = bayes_params_df.loc[bayes_params_df["loss"].idxmin(),"num_iterations"]
        if parameter_name in bayes_params_df.columns.tolist():
        
            # linear regression of parameter (or loss) over iterations
            slope, _, _, p_value, _ = scipy.stats.linregress(
                x = bayes_params_df["num_iterations"], 
                y = bayes_params_df[parameter_name].astype(np.float)
            )
            
            fig, ax = plt.subplots()
            sns.regplot(x = "num_iterations", y = parameter_name, data = bayes_params_df, ax = ax)
            ax.axvline(best_iteration, label = "Best Iteration", linestyle = "--", color = "green", alpha = 0.8)
            ax.set_title("{} over Iterations\nSlope = {:.2f}\nP-Value of Slope = {:.2f}%".format(parameter_name, slope, p_value*100))
            ax.legend()
            return fig, ax
        else:
            print("{} not found among hyperparameters in bayes_params_df".format(parameter_name))

    def plot_all_diagnostics(self, n_samples = 1000):

        figures = {} # {name: (fig, ax)}

        figures["loss_over_iterations"] = self.plot_param_over_iterations("loss")

        # TODO: implement
        if self.hpo.model.model_type == "elastic_net":
            figures["alpha_over_iterations"] = self.plot_param_over_iterations(parameter_name = "alpha")
            figures["alpha_density"] = self.plot_bayes_hyperparam_density(
                parameter_name = "alpha", 
                n_samples = n_samples
            )


            linear_hyperparam_df = self.get_hyperparam_summary()
            linear_hyperparam_df["penalty_type"] = linear_hyperparam_df["l1_ratio"].apply(linear_penalty_type)
            penalty_type_summary_df = linear_hyperparam_df.loc[:,["penalty_type","loss"]].groupby("penalty_type").agg([np.mean, np.std])

            penalty_type_summary_df = penalty_type_summary_df["loss"]

            hist_fig, hist_ax = plt.subplots()
            elastic_net_only_df = linear_hyperparam_df.loc[linear_hyperparam_df["penalty_type"] == "elastic_net",:]
            elastic_net_only_df["l1_ratio"].hist(ax = hist_ax)
            figures["l1_ratio_histogram"] = (hist_fig, hist_ax)

        elif self.hpo.model.model_type == "random_forest":
            for parameter_name in ["min_samples_split", "min_samples_leaf", "max_features"]:

                figures[parameter_name + "_over_iterations"] = self.plot_param_over_iterations(parameter_name = parameter_name)
                figures[parameter_name + "_density"] = self.plot_bayes_hyperparam_density(
                    parameter_name = parameter_name,
                    n_samples = n_samples
                )

        elif self.hpo.model.model_type == "lightgbm":
            for parameter_name in ["num_leaves", "learning_rate", "subsample_for_bin",
                                  "min_child_samples", "reg_alpha", "reg_lambda", "colsample_bytree"]:
                figures[parameter_name + "_over_iterations"] = self.plot_param_over_iterations(parameter_name = parameter_name)
                figures[parameter_name + "_density"] = self.plot_bayes_hyperparam_density(
                    parameter_name = parameter_name,
                    n_samples = n_samples
                )

        elif self.hpo.model.model_type == "xgboost":
            for parameter_name in ["min_child_weight","reg_lambda","colsample_bytree", "gamma"]:
                figures[parameter_name + "_over_iterations"] = self.plot_param_over_iterations(parameter_name = parameter_name)
                figures[parameter_name + "_density"] = self.plot_bayes_hyperparam_density(
                    parameter_name = parameter_name,
                    n_samples = n_samples
                )

        return figures