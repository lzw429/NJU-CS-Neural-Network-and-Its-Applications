import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.X = torch.tensor(data[:, :-1], dtype=torch.float)
        # self.Y = torch.tensor(np.squeeze(np.eye(10)[data[:, -1:].astype(int)]), dtype=torch.float)
        self.Y = torch.tensor(np.squeeze(data[:, -1:].astype(int)), dtype=torch.long)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)
