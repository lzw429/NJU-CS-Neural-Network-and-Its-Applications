import numpy as np
import copy


class ActivationFunc():
    def activate(self, x):
        pass

    def grad(self, x):
        pass


class ReLU(ActivationFunc):
    def activate(self, x):
        return np.maximum(0.0, x)

    def grad(self, x):
        res = copy.deepcopy(x)
        res[np.where(x >= 0.0)] = 1.0
        res[np.where(x < 0.0)] = 0.0
        return res


class LeakyReLU(ActivationFunc):
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def activate(self, x):
        res = copy.deepcopy(x)
        res[np.where(x < 0.0)] = self.alpha * x[np.where(x < 0.0)]
        return res

    def grad(self, x):
        res = copy.deepcopy(x)
        res[np.where(x >= 0.0)] = 1.0
        res[np.where(x < 0.0)] = self.alpha
        return res


class Tanh(ActivationFunc):
    def activate(self, x):
        return np.tanh(x)

    def grad(self, x):
        return 1 - np.power(np.tanh(x), 2)


class Sigmoid(ActivationFunc):
    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def grad(self, x):
        return self.activate(x) * (1 - self.activate(x))


class Linear(ActivationFunc):
    def __init__(self, k=1):
        self.k = k

    def activate(self, x):
        return self.k * x

    def grad(self, x):
        return x * 0 + self.k
