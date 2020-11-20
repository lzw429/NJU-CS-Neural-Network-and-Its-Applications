import numpy as np


class LossFunc():
    def loss(self, y_pred, Y):
        pass

    def grad(self, y_pred, Y):
        pass


class MSELoss(LossFunc):
    def loss(self, y_pred, Y):
        return np.mean((y_pred - Y) ** 2) / 2

    def grad(self, y_pred, Y):
        return np.mean(y_pred - Y)


class LogCoshLoss(LossFunc):
    def loss(self, y_pred, Y):
        return np.sum(np.log(np.cosh(y_pred - Y)))

    def grad(self, y_pred, Y):
        return np.sum(np.tanh(y_pred - Y))
