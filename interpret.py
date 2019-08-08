import pandas as pd

class ModelInterpreter:
	def __init__(self, model):
		self.model = model
		self.get_feature_importance()

	def get_feature_importance(self):
		if self.model.model_type == "random_forest":
			self.feature_importance_df = pd.DataFrame(
				data = {"feature_importance": self.model.model.feature_importances_}, 
				index = self.model.feature_names
			)
		elif self.model.model_type == "elastic_net":
			self.feature_importance_df = pd.DataFrame(
				# elastic net comes from a model pipeline
				data = {"feature_importance": self.model.model.steps[1][1].coef_}, 
				index = self.model.feature_names
			)
		elif (self.model.model_type == "lightgbm") | (self.model.model_type == "lightgbm_special"):
			self.feature_importance_df = pd.DataFrame(
				data = {"feature_importance": self.model.model.feature_importance()}, 
				index = self.model.feature_names
			)
		elif self.model.model_type == "xgboost":
			self.feature_importance_df = pd.DataFrame.from_dict(
				{"feature_importance": self.model.model.get_score(importance_type = "gain")}
			)
		self.feature_importance_df = self.feature_importance_df.sort_values(by = "feature_importance",
																			ascending = False)