# funções para calcular o intervalo de predição
import numpy as np

# from jupyter notebook
MSE = 2115939618.328423
def fit_se(new_data, model, X):
    X = model[:-1].transform(X)
    X = X.todense()
    if X.shape[1] < new_data.shape[1]:
        raise ValueError("number of features of X must be <= the number of features of new_data")
    X_h = np.concatenate((new_data, np.zeros((1, X.shape[1] - new_data.shape[1]))), axis=1).T

    inverse = np.linalg.inv

    return np.sqrt(MSE * (X_h.T @ inverse(X.T @ X) @ X_h)).item()

def pred_se(new_data, model, X):
    se_yhat_new = fit_se(new_data, model, X)

    return np.sqrt(MSE + (se_yhat_new)**2)

def get_interval(prediction_value, new_data, model, X, confidence=0.95):

    new_data_transformed = model[:-1].transform(new_data).todense()
    se = pred_se(new_data_transformed, model, X)
    
    t_value = 1.645

    lower, upper = prediction_value - se*t_value, prediction_value + se*t_value

    return lower, upper

def predict_with_interval(new_data, model, X):
    prediction_value = model.predict(new_data).item()
    lower, upper = get_interval(prediction_value, new_data, model, X)

    return dict(prediction_value=prediction_value, lower=lower, upper=upper)