import numpy as np


class Optimizer():
    def update_parameters(self, parameters, grads):
        pass


class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def update_parameters(self, parameters, grads):
        parameters -= self.lr * grads


class Adam(Optimizer):
    def __init__(self, lr):
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-08
        self.m = None
        self.v = None
        self.t = 0
        self.lamda = 0.0  # L2 regularization

    def update_parameters(self, parameters, grads):
        if self.m is None:
            self.m = np.zeros_like(parameters)
        if self.v is None:
            self.v = np.zeros_like(parameters)

        self.t += 1

        grads += self.lamda * parameters
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)

        alpha = self.lr * np.sqrt(1 - np.power(self.beta2, self.t))
        alpha = alpha / (1 - np.power(self.beta1, self.t))

        parameters -= alpha * self.m / (np.sqrt(self.v) + self.eps)


class AdamW(Optimizer):
    def __init__(self, lr):
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-08
        self.m = None
        self.v = None
        self.t = 0
        self.lamda = 0.0  # decoupled weight decay

    def update_parameters(self, parameters, grads):
        if self.m is None:
            self.m = np.zeros_like(parameters)
        if self.v is None:
            self.v = np.zeros_like(parameters)

        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)

        alpha = self.lr * np.sqrt(1 - np.power(self.beta2, self.t))
        alpha = alpha / (1 - np.power(self.beta1, self.t))

        parameters -= (alpha * self.m / (np.sqrt(self.v) + self.eps) + self.lamda * parameters)
