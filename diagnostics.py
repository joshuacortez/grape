import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hyperopt
import scipy.stats

from sklearn.model_selection import KFold

from param_space import parameter_space
from utils import str_to_dict, linear_penalty_type, eval_func_all_dict, eval_func_is_higher_better

import warnings

RANDOM_SEED_ARGNAMES = ["random_state", "seed"]

class ModelDiagnoser:
    """
    Diagnose a model's predictive performance in one go

    Parameters
    ----------
    model : a GRAPE RegressionModel
        Must be a trained model
    train_valid_folds : list of lists of ints or cross validation generator, or None
        optional (default=None)
        The folds used for cross validation
        If list of list of indices, each sublist should contain the indices per fold
        If None, cross validation diagnostics are not computed
    X_test : dataframe of shape [n_samples_test, n_features], or None
        optional (default=None)
        The independent variables in the test set
        Must be supplied along with y_test
        If None, test set diagnostics is not computed
    y_test : list-like (numpy array or pandas series) of shape [n_samples_test], or None
        optional (default=None)
        The dependent variable in the test set
        Must be supplied along with X_test
        If None, test set diagnostics is not computed
    sample_weight_test : list-like (numpy array or pandas series) of shape [n_samples_test], or None
        optional (default=None)
        Sample weights. If None, then samples are equally weighted
        If not None, X_test and y_test must NOT be None
    eval_func_names : str of list of str, optional (default="r_squared")
        The evaluation function used to measure model error
        Model error as training, cross validation, or test error
        The options are: ["r_squared", "mse", "rmse", "mae"]

    Attributes
    ----------
    Attributes which are saved initialization parameters:
        model : a GRAPE RegressionModel
        train_valid_folds : list or cross validation generator
        X_test : dataframe
        y_test : list-like
        sample_weight_test : list-like
        eval_func_names : str of list of str
    
    Derived attributes:
        model_diagnostics : dict 
            Model diagnostics summary
            Contains the following, per evaluation function: 
                training error
                cv error averaged across folds
                cv error's standard deviation across folds
                out-of-bag error, when available
                test error
            Example : {"train-mse":90.1, "cv-mse-mean":83.5, 
            "cv-mse-std":3.75, "oob-mse":81.26, "test-mse":80.01}

    Example
    -------
    see model_diagnoser_test in tests.py
    """

    def __init__(self, model, 
                     train_valid_folds = None,
                     X_test = None,
                     y_test = None,
                     sample_weight_test = None,
                     eval_func_names = "r_squared"):

        self.model = model
        self.train_valid_folds = train_valid_folds
        self.X_test = X_test
        self.y_test = y_test
        self.sample_weight_test = sample_weight_test
        if (sample_weight_test is None) and (model.sample_weight is not None):
            warnings.warn("sample_weight_test is set to None despite the model was trained with a sample_weight. Double check if you really want to do this.")
        self.eval_func_names = eval_func_names
        
        assert hasattr(model, "model"), "RegressionModel is still untrained. Fit model first before diagnosing error"
        assert (self.X_test is None) == (self.y_test is None), "X_test and y_test arguments have to be both supplied or neither"
        if self.sample_weight_test is not None:
            assert (self.X_test is not None) and (self.y_test is not None), "Cannot supply sample_weight_test without a test_set"

    def get_model_diagnostics(self, eval_func_names):
        """
        Computes model diagnostics based on provided evaluation functions
        """
        if isinstance(eval_func_names, str):
            eval_func_names = [eval_func_names]

        self.model_diagnostics = {}

        train_y_true = self.model.y_train
        train_y_pred = self.model.predict(self.model.X_train)

        if (self.X_test is not None) & (self.y_test is not None):
            test_y_true = self.y_test
            test_y_pred = self.model.predict(self.X_test)

        for eval_func_name in eval_func_names:
            eval_func = eval_func_all_dict["sklearn_metric"][eval_func_name]
            self.model_diagnostics["train-{}".format(eval_func_name)] = eval_func(y_true = train_y_true, 
                                                                                    y_pred = train_y_pred,
                                                                                    sample_weight = self.model.sample_weight)
            if self.model.model_type == "random_forest":                                                                  
                oob_y_pred = self.model.model.oob_prediction_

                self.model_diagnostics["oob-{}".format(eval_func_name)] = eval_func(y_true = train_y_true,
                                                                                    y_pred = oob_y_pred,
                                                                                    sample_weight = self.model.sample_weight)

            if (self.X_test is not None) & (self.y_test is not None):
                self.model_diagnostics["test-{}".format(eval_func_name)] = eval_func(y_true = test_y_true, 
                                                                                    y_pred = test_y_pred,
                                                                                    sample_weight = self.sample_weight_test)

        if self.train_valid_folds is not None:
            model_params = {key:val for key,val in self.model.model_params.items() if key not in RANDOM_SEED_ARGNAMES}
            cv_scores = self.model.cross_validate(train_valid_folds = self.train_valid_folds,
                                                  eval_func_names = eval_func_names,
                                                 model_params = model_params)

            for eval_func_name in eval_func_names:         
                eval_func_mean = cv_scores["cv-{}-mean".format(eval_func_name)]
                eval_func_std = cv_scores["cv-{}-std".format(eval_func_name)]
                if self.model.model_type in ["lightgbm", "xgboost"]:
                    if eval_func_is_higher_better[eval_func_name]:
                        best_idx = np.argmax(eval_func_mean)
                    else:
                        best_idx = np.argmin(eval_func_mean)
                    eval_func_mean = eval_func_mean[best_idx]
                    eval_func_std = eval_func_std[best_idx]
                
                self.model_diagnostics["cv-{}-mean".format(eval_func_name)] = eval_func_mean
                self.model_diagnostics["cv-{}-std".format(eval_func_name)] = eval_func_std

    
    def plot_actual_vs_pred(self, X, y, figsize = (6,6), title_add_text = ""):
        """
        Graphs an actual versus predicted scatterplot
        """
        y_pred = self.model.predict(X)

        max_val = max(max(y_pred), max(y))
        min_val = min(min(y_pred), min(y))
        equality_line = np.linspace(start = min_val, stop = max_val, num = 100)

        fig, ax = plt.subplots(figsize = figsize)
        ax.scatter(x = y, y = y_pred, alpha = 0.75)
        ax.plot(equality_line, equality_line, linestyle = "--", color = "grey", alpha = 0.75)
        
        title_text = "Actual vs Predicted"
        if title_add_text:
            title_text = "{} ({})".format(title_text, title_add_text)
        
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(title_text)

        return fig, ax

    def show_all_diagnostics(self):
        """
        Convenience function for showing all model diagnostics
        Prints model diagnostic measures
        Plots actual versus predicted plot for training set and test set (if available)
        """
        self.get_model_diagnostics(self.eval_func_names)
        print(self.model_diagnostics)

        figures = {}
        figures["training_set"] = self.plot_actual_vs_pred(X = self.model.X_train, 
                                         y = self.model.y_train,
                                        title_add_text = "Training Set")

        if (self.X_test is not None) and (self.y_test is not None):
            figures["test_set"] = self.plot_actual_vs_pred(X = self.X_test, 
                                        y = self.y_test,
                                        title_add_text = "Test Set")
        
        return figures

class HPODiagnoser:
    """
    Diagnose the SMBO hyperparameter optimization by asesssing graphs

    Parameters
    ----------
    hpo : a GRAPE HyperParameterOptimizer
        Must have already tuned a model

    Example
    -------
    see hpo_diagnoser_test in tests.py
    """

    def __init__(self, hpo):
        self.hpo = hpo

        assert hasattr(hpo, "hyperopt_summary"), "HyperParameterOptimizer still hasn't tuned a model. Tune a model first before diagnosing."

    def get_hyperparam_summary(self):
        """
        Creates a dataframe of parameter settings per iteration of the hpo
        """
        # Create a new empty dataframe for storing parameters per iteration
        bayes_params_df = pd.DataFrame(
            columns = self.hpo.best_params.keys(),
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
        """
        Per tuned hyperparameter, plots the prior versus the empirical density from the HPO's iterations
        This plot is useful for determining whether or not to adjust the prior.
            1. If the empirical density from the HPO is trying to "push past" 
            the bounds of the prior, consider widening the bounds of the prior.
            2. If the empirical density from the HPO is very narrow versus
            the spread of the prior, consider narrowing the bounds of the prior.

        Parameters
        ----------
        parameter_name : str
            Name of the hyperparameter that was among those tuned
            The hyperparameter must be among those defined in parameter_space_dict

        n_samples : int, optional (default=1000)
            Number of samples for sampling from the hyperparameter's prior
            The samples are used to visualize the prior
        """

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
                self.hpo.best_params[parameter_name], 
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

    def plot_hyperparam_iterations(self, parameter_name):
        """
        Scatterplot of hyperparameter values (y-axis) versus iteration in HPO (x-axis)
        Includes a linear regression to see trend of hyperparameter over the iterations

        Parameters:
        ----------
        parameter_name : str
            Name of the hyperparameter that was among those tuned
            The hyperparameter must be among those defined in parameter_space_dict
        """
        bayes_params_df = self.get_hyperparam_summary()
        best_iteration = bayes_params_df.loc[bayes_params_df["loss"].idxmin(),"n_iterations"]
        if parameter_name in bayes_params_df.columns.tolist():
        
            # linear regression of parameter (or loss) over iterations
            slope, _, _, p_value, _ = scipy.stats.linregress(
                x = bayes_params_df["n_iterations"], 
                y = bayes_params_df[parameter_name].astype(np.float)
            )
            
            fig, ax = plt.subplots()
            sns.regplot(x = "n_iterations", y = parameter_name, data = bayes_params_df, ax = ax)
            ax.axvline(best_iteration, label = "Best Iteration", linestyle = "--", color = "green", alpha = 0.8)
            ax.set_title("{} over Iterations\nSlope = {:.2f}\nP-Value of Slope = {:.2f}%".format(parameter_name, slope, p_value*100))
            ax.legend()
            return fig, ax
        else:
            print("{} not found among hyperparameters in bayes_params_df".format(parameter_name))

    def show_all_diagnostics(self, n_samples = 1000):
        """
        Convenience function that loops over each hyperparameter and plots the following:
            plot_bayes_hyperparam_density
            plot_hyperparam_iterations

        Also plots the HPO's loss over iterations

        Parameters
        ----------
        n_samples : int, optional (default=1000)
            Number of samples for sampling from the hyperparameter's prior
            The samples are used to visualize the prior

        Returns 
        -------
        figures : dict
            Contains matplotlib figures as the dict values
        """

        figures = {} # {name: (fig, ax)}

        figures["loss_iterations"] = self.plot_hyperparam_iterations("loss")

        if self.hpo.model.model_type == "elastic_net":
            figures["alpha_iterations"] = self.plot_hyperparam_iterations(parameter_name = "alpha")
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

        else:
            for parameter_name in parameter_space[self.hpo.model.model_type]:

                figures[parameter_name + "_iterations"] = self.plot_hyperparam_iterations(parameter_name = parameter_name)
                figures[parameter_name + "_density"] = self.plot_bayes_hyperparam_density(
                    parameter_name = parameter_name,
                    n_samples = n_samples)

        return figures