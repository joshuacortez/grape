import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hyperopt
import scipy.stats

from param_space import parameter_space
from utils import str_to_dict, linear_penalty_type

class HPODiagnoser:

    def __init__(self, hpo):
        self.hpo = hpo
        self._model_type = self.hpo._model._model_type

        self.df_optimization_summary = pd.DataFrame(self.hpo.optimization_summary)

        # find the index of minimum loss
        min_loss_idx = self.df_optimization_summary["loss"].idxmin()
        # get the parameter dictionary from the iteration with the lowest loss
        best_params_dict =  str_to_dict(self.df_optimization_summary.loc[min_loss_idx,"params"])
        
        if self._model_type in ["lightgbm", "lightgbm_special"]:
            # include number of boosting rounds from hyperparameter optimization
            best_params_dict["num_iterations"] = self.df_optimization_summary.loc[min_loss_idx,"estimators"]
        if self._model_type == "xgboost":
            best_params_dict["num_boost_round"] = self.df_optimization_summary.loc[min_loss_idx,"num_boost_round"]
        self.best_params_dict_ = best_params_dict
        self.best_loss_ = self.df_optimization_summary.loc[min_loss_idx,"loss"]

    def get_hyperparam_summary(self):
        # Create a new empty dataframe for storing parameters
        columns = list(str_to_dict(
            self.df_optimization_summary.loc[0, 'params']
        ).keys())
        
        bayes_params_df = pd.DataFrame(
            columns = columns,
            index = list(range(len(self.df_optimization_summary)))
        )
        # Add the results with each parameter a different column
        for i, params in enumerate(self.df_optimization_summary['params']):
            bayes_params_df.loc[i, :] = list(str_to_dict(params).values())

        for colname in self.df_optimization_summary.columns:
            if colname != "params":
                bayes_params_df[colname] = self.df_optimization_summary[colname]

        return bayes_params_df

    def plot_bayes_hyperparam_density(self, parameter_name, n_samples = 1000):
        bayes_params_df = self.get_hyperparam_summary()
        parameter_space_dict = parameter_space[self._model_type]
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
        if parameter_name in bayes_params_df.columns:
        
            # linear regression of parameter (or loss) over iterations
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
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

    def get_all_diagnostics(self, n_samples = 1000):

        figures = {} # {name: (fig, ax)}

        # TODO: implement