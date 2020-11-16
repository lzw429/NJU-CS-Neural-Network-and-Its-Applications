import numpy as np


class LossFunc():
    def loss(self, y_pred, Y):
        pass

    def grad(self, y_pred, Y):
        pass


class MSELoss(LossFunc):
    def loss(self, y_pred, Y):
        return np.sum((Y - y_pred) ** 2)

    def grad(self, y_pred, Y):
        return 2 * np.sum(Y - y_pred)


def Huber(y_pred, Y, delta):
    loss = np.where(np.abs(Y - y_pred) < delta, 0.5 * ((Y - y_pred) ** 2),
                    delta * np.abs(Y - y_pred) - 0.5 * (delta ** 2))
    return np.sum(loss)


def Log_cosh(y_pred, Y):
    loss = np.log(np.cosh(y_pred - Y))
    return np.sum(loss)


def MAE(y_pred, Y):
    return np.sum(np.abs(Y - y_pred))
