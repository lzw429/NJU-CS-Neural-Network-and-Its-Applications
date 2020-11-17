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
        x[np.where(x >= 0)] = 1.0
        x[np.where(x < 0)] = 0.0
        return x


class Tanh(ActivationFunc):
    def activate(self, x):
        return np.tanh(x)

    def grad(self, x):
        return 1 - np.power(np.tanh(x), 2)
