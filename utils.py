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
