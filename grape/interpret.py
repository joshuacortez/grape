import pandas as pd

# todo: use rfpimp for feature importance instead
# todo: use SHAP

class ModelInterpreter:
	"""
	Interpret the regression model using feature importance and SHAP

	Parameters
	----------
	model : a GRAPE RegressionModel
		Must be a trained model

	Attributes
	----------
	feature_importance_df : dataframe of shape [n_features, 2]
	"""
	def __init__(self, model):
		assert hasattr(model, "model"), "RegressionModel is still untrained. Fit model first before interpreting it"
		self.model = model
		self.get_feature_importance()

	def get_feature_importance(self):
		if self.model.model_type == "random_forest":
			self.feature_importance_df = pd.DataFrame(
				data = {"feature_importance": self.model.model.feature_importances_}, 
				index = self.model.feature_names
			)
		elif self.model.model_type == "elastic_net":
			if self.model.log_target_reg:
				feature_importance = self.model.model.regressor_.steps[1][1].coef_
			else:
				feature_importance = self.model.model.steps[1][1].coef_

			self.feature_importance_df = pd.DataFrame(
				# elastic net comes from a model pipeline
				data = {"feature_importance": feature_importance}, 
				index = self.model.feature_names
			)
		elif self.model.model_type == "lightgbm":
			self.feature_importance_df = pd.DataFrame(
				data = {"feature_importance": self.model.model.feature_importances_}, 
				index = self.model.feature_names
			)
		elif self.model.model_type == "xgboost":
			self.feature_importance_df = pd.DataFrame.from_dict(
				{"feature_importance": self.model.model.feature_importances_}
			)
		self.feature_importance_df = self.feature_importance_df.sort_values(by = "feature_importance",
																			ascending = False)