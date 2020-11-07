import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data


def fitted_func(x):
    return x * x + 2 * x + 1


class Dataset(torch.utils.data.Dataset):

    def __init__(self, x):
        self.inputs = []
        self.golden = []
        for i in x:
            self.inputs.append([i, i * i])
            self.golden.append([fitted_func(i)])
        self.inputs = torch.tensor(self.inputs, dtype=torch.float)
        self.golden = torch.tensor(self.golden, dtype=torch.float)

    def __getitem__(self, index):
        return self.inputs[index], self.golden[index]

    def __len__(self):
        return len(self.inputs)


class Neuron(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.activate = nn.LeakyReLU()

    def forward(self, x):
        return self.activate(self.linear(x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epoch", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--log_path", type=str,
                        default="C:\\Users\\Dell\\Documents\\GitHub\\NJU-CS-Neural-Network-and-Its-Applications\\homework_4\\log.txt")
    args = parser.parse_args()

    log_file = open(args.log_path, "w")
    neuron = Neuron()
    x = np.arange(-5, 5, 0.01)
    dataset = Dataset(x)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = optim.AdamW(neuron.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # draw the picture before training
    plt.plot(x, dataset.golden, color='b')
    plt.show()

    train_finished = False
    for epoch_idx in range(args.num_epoch):
        running_loss = 0.0
        if train_finished == True:
            break
        for batch_idx, sample_batched in enumerate(dataloader):
            inputs, golden = sample_batched

            optimizer.zero_grad()
            outputs = neuron(inputs)
            loss = criterion(outputs, golden)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch_idx + 1, batch_idx + 1, running_loss), file=log_file)
            if loss < 0.000001:
                train_finished = True
                break
            running_loss = 0.0

    print('[INFO] Finished Training')

    # draw the picture after training
    inputs_list = []
    outputs_list = []
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader):
            inputs, golden = sample_batched
            outputs = neuron(inputs)
    # plt.plot()
    # plt.show()
