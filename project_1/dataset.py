class Dataset:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # prepare all batches
        self.batch_list = []


    def sample_batch(self, batch_idx):
        if batch_idx < len(self.batch_list):
            return self.batch_list[batch_idx]
        raise IndexError
