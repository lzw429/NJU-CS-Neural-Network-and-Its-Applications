import numpy as np
import torch
import torch.utils.data


def sample_generate():
    x = np.arange(0, 6.3, 0.01)
    inputs = []
    golden = []
    for i in x:
        inputs.append(i)
        golden.append(np.sin(i))
    return inputs, golden


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float)
        self.Y = torch.tensor(Y, dtype=torch.float)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)
