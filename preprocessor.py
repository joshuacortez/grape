from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import pandas as pd

# todo: other feature engineering tasks
    # automatically removing columns that have only 1 unique value
    # automatically assigning a new column that says whether or not a feature had missing values


class FeaturePreprocessor:
    """
    Preprocessing steps for features
    Currently supported is categorical encoding

    Parameters
    ----------
    cat_encoding_type : str
        Either label_encoder (generates a label per category) or label_binarizer (dummy variables)
    
    Attributes
    ----------
    Attributes which are saved initialization parameters:
        cat_encoding_type : str

    Derived attributes:
        col_encoder_dict : dict
            Keys correspond to column names in X_df  
            Values correspond to LabelEncoder or LabelBinarizer objects

    """
    def __init__(self, cat_encoding_type):
        assert cat_encoding_type in ["label_encoder", "label_binarizer"], "{} cat_encoding_type is not supported"
        self.cat_encoding_type = cat_encoding_type

    def fit(self, X_df, cat_cols_list):
        """
        Fits LabelEncoder or LabelBinarizer objects for all categorical variables in a dataframe

        Parameters
        ----------
        X_df : dataframe
        cat_cols_list : list of str
            Contains the column names in X_df that are categorical and should be transformed
        """
        X_df = X_df.copy()
        
        assert len(cat_cols_list) > 1, "Need to specify at least one categorical variable"
        for col in cat_cols_list:
            assert col in X_df.columns, "{} not found among columns in X_df".format(col)
        self._cat_cols_list = cat_cols_list
        
        self.col_encoder_dict = {}
        
        if self.cat_encoding_type == "label_encoder":
            
            # tree based methods only need to preprocess categorical variables
            if self._cat_cols_list:

                # label encoding on categorical variables
                for col in self._cat_cols_list:
                    labelencoder = LabelEncoder()
                    labelencoder.fit(X_df[col])
                    self.col_encoder_dict[col] = labelencoder
            
        if self.cat_encoding_type == "label_binarizer":
            
            # one hot encoding for categorical variables
            for col in self._cat_cols_list:

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
                
    def transform(self, X_df):
        """
        Transforms the categorical variables

        Parameters
        ----------
        X_df : dataframe

        Returns
        -------
        X_df : dataframe
            Transformed after categorical encoding
        """
        X_df = self._encode_categorical_cols(X_df)

        return X_df
    
    def fit_transform(self, X_df, cat_cols_list):
        """
        Convenience function for fitting and transforming in one go

        See fit and transform methods

        Parameters:
        -----------
        X_df : dataframe
        cat_cols_list : list of str

        Returns:
        --------
        X_df : dataframe
            Transformed after categorical encoding
        """
        self.fit(X_df, cat_cols_list)
        
        X_df = self.transform(X_df)
        
        return X_df

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