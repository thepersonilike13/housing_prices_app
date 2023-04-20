# funções para calcular o intervalo de predição
import numpy as np

# from jupyter notebook
rmse = 47527.36
mse = rmse **2

def fit_std_error(new_data, model, X_train):
    X = model[:-1].transform(X_train).todense()
    if X.shape[1] < new_data.shape[1]:
        raise ValueError("number of features of X_train must be <= the number of features of new_data")
    X_h = np.concatenate((new_data, np.zeros(X.shape[1] - new_data.shape[1]).reshape(1, -1)), axis=1).T

    inverse = np.linalg.inv

    return np.sqrt(mse * (X_h.T @ inverse(X.T @ X) @ X_h)).item()

def prediction_std_error(new_data, model, X_train):
    se_yhat = fit_std_error(new_data, model, X_train)

    return np.sqrt(mse + (se_yhat)**2)

def get_interval(prediction_value, new_data, model, X_train, confidence=0.95):

    new_data_transformed = model[:-1].transform(new_data).todense()
    se = prediction_std_error(new_data_transformed, model, X_train)
    
    n_predictors = len(X_train.columns)
    t_value = 1.645

    lower, upper = prediction_value - se*t_value, prediction_value + se*t_value

    return lower, upper

def predict_with_interval(new_data, model, X_train):
    prediction_value = model.predict(new_data).item()
    lower, upper = get_interval(prediction_value, new_data, model, X_train)

    return dict(prediction_value=prediction_value, lower=lower, upper=upper)