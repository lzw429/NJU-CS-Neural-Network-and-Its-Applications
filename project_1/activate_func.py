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
        x[np.where(x >= 0)] = 1
        x[np.where(x < 0)] = 0
        return x
