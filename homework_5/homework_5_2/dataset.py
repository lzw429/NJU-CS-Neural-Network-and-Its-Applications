import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.X = data[:, :-1]
        self.Y = torch.squeeze(np.eye(10)[data[:, -1:].astype(int)])
        print(1)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)
