import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# x1 = np.arange(-5, 5, 0.1)
# x2 = np.arange(-5, 5, 0.1)
# sample_inputs = []
# sample_golden = []
# for i in x1:
#     for j in x2:
#         sample_inputs.append((i, j))  # the inputs: 2-d
#         sample_golden.append(fitted_func(i, j))  # the golden outputs: 1-d
# return np.array(sample_inputs), np.array(sample_golden)

def radius_square(x, y):
    return x * x + y * y


def sample_generate():
    x = np.arange(-2, 2, 0.01)
    y = np.arange(-2, 2, 0.01)

    sample_inputs = []
    sample_golden = []
    sample_0 = []
    sample_1 = []
    for i in x:
        for j in y:
            if radius_square(i, j) <= np.sqrt(2):
                sample_inputs.append((i, j))
                sample_golden.append(0)
            else:
                sample_inputs.append((i, j))
                sample_golden.append(1)

    return np.array(sample_inputs), np.array(sample_golden)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, Y):
        self.inputs = torch.tensor(X, dtype=torch.float)
        self.golden = torch.tensor(Y, dtype=torch.float)

    def __getitem__(self, index):
        return self.inputs[index], self.golden[index]

    def __len__(self):
        return len(self.inputs)
