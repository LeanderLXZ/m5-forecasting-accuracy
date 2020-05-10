import numpy as np
from sklearn.metrics import  mean_absolute_error

def eval_func(y_true, y_pred):
    mse = mean_absolute_error(y_true, y_pred)
    n = len(y_pred)
    s = 0
    for i in range(1,len(y_pred)):
        s += (y_true[i] - y_true[i-1]) ** 2
    a = 1 / (len(y_pred) - 1) * s
    return np.sqrt((1 / n) * mse / a)
