import numpy as np


class Dataset:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def shuffle(self):
        np.random.shuffle(self.X)
        np.random.shuffle(self.Y)

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def getitem(self, index):
        if index > len(self.X) or index > len(self.Y):
            return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # shuffle the data
        if shuffle:
            self.dataset.shuffle()

        # prepare all batches
        self.batch_list = []
        for batch_idx in range(len(dataset) // batch_size):
            begin = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size
            self.batch_list.append((self.dataset.X[begin: end], self.dataset.Y[begin: end]))

        if not drop_last:
            begin = (len(dataset) // batch_size) * batch_size
            end = len(dataset)
            self.batch_list.append((self.dataset.X[begin:end], self.dataset.Y[begin:end]))

    def get_num_batch(self):
        return len(self.batch_list)

    def get_batch(self, batch_idx):
        if batch_idx < len(self.batch_list):
            return self.batch_list[batch_idx]
        raise IndexError
