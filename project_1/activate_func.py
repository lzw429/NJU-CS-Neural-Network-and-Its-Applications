import numpy as np


class ActivationFunc():
    def activate(self, x):
        pass

    def grad(self, x):
        pass


class ReLU(ActivationFunc):
    def activate(self, x):
        return np.maximum(0.0, x)

    def grad(self, x):
        x[np.where(x >= 0.0)] = 1.0
        x[np.where(x < 0.0)] = 0.0
        return x


class LeakyReLU(ActivationFunc):
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def activate(self, x):
        x[np.where(x < 0.0)] = self.alpha * x[np.where(x < 0.0)]
        return x

    def grad(self, x):
        x[np.where(x >= 0.0)] = 1.0
        x[np.where(x < 0.0)] = self.alpha
        return x


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
