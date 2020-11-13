import numpy as np


def Huber(y_pred, Y, delta):
    loss = np.where(np.abs(Y - y_pred) < delta, 0.5 * ((Y - y_pred) ** 2),
                    delta * np.abs(Y - y_pred) - 0.5 * (delta ** 2))
    return np.sum(loss)


def Log_cosh(y_pred, Y):
    loss = np.log(np.cosh(y_pred - Y))
    return np.sum(loss)


def MSE(y_pred, Y):
    return np.sum((Y - y_pred) ** 2)


def MAE(y_pred, Y):
    return np.sum(np.abs(Y - y_pred))
