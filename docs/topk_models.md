Two functions: topk_performance(...) and best_topk_model(...)


def topk_performance(model,n_topk,x_train,y_train,x_test,y_test,yencoder):

This function computes Top-k performance results based on the given model and the train and test data. The function trains on the x data and the target value, and then runs the model on the test data. It selects k possible matches which have the highest probability of being a match. It then computes the positive prediction value (PPV) for each class, from worst to best, as well as the Top-k accuracy, where the model is considered a success if the the target class is one of the k matches selected by the model. Therefore, Top-k accuracy will have the highest probabiity because it includes all three classes. PPVs are calculated by by adding all the instances where the prediction was correct divided by the total number of predictions.

Example: see solar_radiation_example1 under topk_classification

    model: model object
    any model that returns prediction probabilities for each class

    n_topk: integer
    number of Top-k classes to predict

    x_train: dataframe
    x data (attributes) to train on

    y_train: dataframe
    target values to train on

    x_test: dataframe
    x data (attributes) to test the model

    y_test: dataframe
    target values to test the model

    yencoder: function
    encoder used to transform target values to classes


Returns: y_pred, p_res, class_sort, prob_sort

    y_pred: numpy.ndarray
    Top-k predictions of target valaues

    p_res: numpy.ndarray
    prediction probabilities for each of the Top-k classes

    class_sort: numpy.ndarray
    Top-k class predictions of target valaues in categorical form

    prob_sort: Top-k accuracy measurements
    PPVs for each class from best to worst as well as Top-k accuracy



def best_topk_model(topk_models,n_topk,x_train,y_train,x_test,y_test,yencoder) 

This function uses the topk_performance function above to compute Top-k accuracy measurements for different models and then selects the best performing model.

Example: see solar_radiation_example1 under topk_classification

    topk_models: model object list
    Models that return prediction probabilities for each class

    n_topk: integer
    number of Top-k classes to predict

    x_train: dataframe
    x data (attributes) to train on

    y_train: dataframe
    target values to train on

    x_test: dataframe
    x data (attributes) to test the model

    y_test: dataframe
    target values to test the model

    yencoder: function
    encoder used to transform target values to classes


Returns: y_pred, p_res, class_sort, prob_sort

    y_pred: numpy.ndarray
    Top-k predictions of target valaues

    p_res: numpy.ndarray
    prediction probabilities for each of the Top-k classes

    class_sort: numpy.ndarray
    Top-k class predictions of target valaues in categorical form

    prob_sort: Top-k accuracy measurements
    PPVs for each class from best to worst as well as Top-k accuracy
