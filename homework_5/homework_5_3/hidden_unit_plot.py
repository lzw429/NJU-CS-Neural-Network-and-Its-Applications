import copy

import numpy as np
import matplotlib.pyplot as plt


def leakyReLU(x):
    res = copy.deepcopy(x)
    res[np.where(x < 0.0)] = 0.1 * x[np.where(x < 0.0)]
    return res


def tanh(x):
    return np.tanh(x)


weight = [5.2373, -0.9132, 0.4150, 3.0864, -1.1195]
bias = [-1.6582, 2.0476, 0.0269, -3.4080, 2.6947]

x = np.array(np.arange(0, 1, 0.01))
for i in range(5):
    y = tanh(weight[i] * x + bias[i])
    plt.plot(x, y)
    plt.show()
