import argparse
from homework_5_1.dataset import sample_generate, Dataset
from homework_5_1.neural_network import NeuralNetwork
from homework_5_1.perceptron import Perceptron
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim


def perceptron_test():
    perceptron = Perceptron()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    optimizer = optim.AdamW(perceptron.parameters(), lr)
    criterion = nn.BCELoss()

    last_loss = 0.0
    for epoch_idx in range(num_epoch):
        running_loss = 0.0
        for batch_idx, sample_batched in enumerate(dataloader):
            inputs, golden = sample_batched

            optimizer.zero_grad()
            outputs = perceptron(inputs)
            loss = criterion(outputs, golden)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print('[%d] loss: %.3f' % (epoch_idx + 1, running_loss))
        if running_loss > last_loss:
            print("[WARN] the loss is increasing")
        last_loss = running_loss
        if running_loss < 0.001:
            break

    print("[INFO] Finished Training")


def neural_network_test():
    neural_network = NeuralNetwork(n_h=2, hidden_size=20)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    optimizer = optim.AdamW(neural_network.parameters(), lr)
    criterion = nn.BCELoss()

    last_loss = 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_epoch", type=int, default=100000)
    args = parser.parse_args()
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    X, Y = sample_generate()

    dataset = Dataset(X, Y)
    perceptron_test()
