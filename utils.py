from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
import pandas as pd

class PredictorPreprocessor:
    def __init__(self, model_type, scale_predictors = True):
        assert model_type in ["random_forest", "elastic_net", "lightgbm", "lightgbm_special", "xgboost"]
        self._model_type = model_type
        self._scale_predictors = scale_predictors
    
    # helper functions that encodes all categorical variables
    def _encode_categorical_cols(self, X_df):
        
        # not sure if copy is really needed
        X_df = X_df.copy()
        
        for col in self.col_encoder_dict.keys():
            encoder = self.col_encoder_dict[col]
            
            # first check if labels in X_df[col] are among the labels in the encoder
            unique_labels = X_df[col].unique().tolist()
            is_subset = set(unique_labels).issubset(set(encoder.classes_.tolist()))
            assert is_subset, "Found labels in {} that are not found among the classes in the encoder {}".format(col, set(unique_labels) - set(encoder.classes_.tolist()) )
            
            if isinstance(encoder, LabelBinarizer):
                onehot_df = encoder.transform(X_df[col])
                onehot_cols = ["{}_{}".format(col, label) for label in encoder.classes_]
                onehot_df = pd.DataFrame(data = onehot_df, columns = onehot_cols, index = X_df.index)

                # drop the first class 
                onehot_cols = onehot_cols[1:]
                onehot_df = onehot_df.loc[:,onehot_cols]
            
                # append the one hot encoded columns to X_df
                X_df = pd.concat([X_df, onehot_df], axis = 1)
                
                # drop the original column
                X_df = X_df.drop(columns = col)
                
            if isinstance(encoder, LabelEncoder):
                X_df[col] = encoder.transform(X_df[col])

        return X_df
    
    def fit(self, X_df, categorical_cols_list):
        X_df = X_df.copy()
        
        assert type(categorical_cols_list) == list
        self._categorical_cols_list = categorical_cols_list
        
        self.col_encoder_dict = {}
        
        if self._model_type in ["random_forest", "lightgbm", "lightgbm_special", "xgboost"]:
            
            # tree based methods only need to preprocess categorical variables
            if self._categorical_cols_list:

                # label encoding on categorical variables
                for col in self._categorical_cols_list:
                    labelencoder = LabelEncoder()
                    labelencoder.fit(X_df[col])
                    self.col_encoder_dict[col] = labelencoder
            
        if self._model_type in ["elastic_net"]:
            
            # one hot encoding for categorical variables
            for col in self._categorical_cols_list:

                # Label Encode those with 2 labels only 
                if X_df[col].nunique() == 2:
                    labelencoder = LabelEncoder()
                    labelencoder.fit(X_df[col])
                    self.col_encoder_dict[col] = labelencoder

                else:
                    onehotencoder = LabelBinarizer()
                    onehotencoder.fit(X_df[col])
                    self.col_encoder_dict[col] = onehotencoder 

            # need to pre-emptively transform the one hot encoded variables before scaling
            # https://stats.stackexchange.com/questions/69568/whether-to-rescale-indicator-binary-dummy-predictors-for-lasso
            X_df = self._encode_categorical_cols(X_df) 
            
        # scaling for all predictor variables
        if self._scale_predictors:
            scaler = StandardScaler(with_mean = True, with_std = True)
            scaler.fit(X_df)
            self.scaler = scaler
                
    def transform(self, X_df):
        X_df = self._encode_categorical_cols(X_df)
                
        if self._scale_predictors:
            X_df = pd.DataFrame(data = self.scaler.transform(X_df), 
                                columns = X_df.columns,
                                index = X_df.index)

        return X_df
    
    def fit_transform(self, X_df, categorical_cols_list):
        self.fit(X_df, categorical_cols_list)
        
        X_df = self.transform(X_df)
        
        return X_df

def get_sample_weight(samples, metric):
    assert metric in ["l1", "l2"]
        
    if metric == "l1":
        # weight the observations so the training can implicitly optimize for Mean Absolute Percentage Error (MAPE) 
        # instead of Mean Absolute Error (MAE)
        sample_weight = samples.apply(lambda y: 1/y)

    if metric == "l2":
        # weight the observations so the training can implicitly optimize for Mean Squared Percentage Error (MSPE) 
        # instead of Mean Squared Error (MSE)
        sample_weight = samples.apply(lambda y: 1/(y**2))
        
    # normalize the weights to sum up to 1.0
    sample_weight = sample_weight/sample_weight.sum()
    
    # turn into numpy array
    sample_weight = sample_weight.get_values()
    
    return sample_weight
