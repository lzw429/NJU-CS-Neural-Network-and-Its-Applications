import numpy as np
import torch


def sample_generate():
    x = np.arange(0, 1, 0.01) #
    inputs = []
    golden = []
    for i in x:
        inputs.append(i)
        golden.append(np.sin(i))
    return inputs, golden


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)
