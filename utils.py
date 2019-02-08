def get_sample_weight(samples, metric):
    '''
    Positional Arguements:
        samples: Numpy Array. Required.
        metric: String. 'l1' or 'l2'. Required.
    '''
    # TODO: Raise an appropriate error instead of AssertionError
    assert metric in ["l1", "l2"]
        
    if metric == "l1":
        # weight the observations so the training can implicitly optimize for Mean Absolute Percentage Error (MAPE) 
        # instead of Mean Absolute Error (MAE)
        sample_weight = samples.apply(lambda y: 1/y)

    elif metric == "l2":
        # weight the observations so the training can implicitly optimize for Mean Squared Percentage Error (MSPE) 
        # instead of Mean Squared Error (MSE)
        sample_weight = samples.apply(lambda y: 1/(y**2))
        
    # normalize the weights to sum up to 1.0
    sample_weight = sample_weight/sample_weight.sum()
    
    # turn into numpy array
    sample_weight = sample_weight.get_values()
    
    return sample_weight

# taken from https://stackoverflow.com/questions/45006341/xgboost-how-to-use-mae-as-objective-function
def _huber_approx_obj(preds, dtrain):
    d = preds - dtrain.get_label() #remove .get_labels() for sklearn
    h = 1  #h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = (1 / scale) / scale_sqrt
    return grad, hess

def get_elastic_net_l1_ratio(model_params):
    penalty_type = model_params["l1_ratio"]["penalty_type"]
    if penalty_type == "lasso":
        l1_ratio = 1
    elif penalty_type == "ridge":
        l1_ratio = 0
    else:
        l1_ratio = model_params["l1_ratio"]["l1_ratio"]

    return l1_ratio