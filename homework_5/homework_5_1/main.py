import argparse
from homework_5_1.dataset import sample_generate, Dataset
from homework_5_1.neural_network import NeuralNetwork
from homework_5_1.perceptron import Perceptron
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim


def perceptron_test():
    print("[INFO] The single-layer Perceptron training starts")
    perceptron = Perceptron()
    model_test(perceptron)


def neural_network_test():
    print("[INFO] The neural network training starts")
    neural_network = NeuralNetwork(n_h=2, hidden_size=20)
    model_test(neural_network)


def model_test(model):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    optimizer = optim.AdamW(model.parameters(), lr)
    criterion = nn.BCELoss()

    last_loss = 0.0
    for epoch_idx in range(num_epoch):
        running_loss = 0.0
        for batch_idx, sample_batched in enumerate(dataloader):
            inputs, golden = sample_batched

            optimizer.zero_grad()
            outputs = model(inputs)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_epoch", type=int, default=100000)
    parser.add_argument("--model", type=int, default=1)
    args = parser.parse_args()
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    test_model = args.model

    X, Y = sample_generate()

    dataset = Dataset(X, Y)
    if test_model == 0:
        perceptron_test()
    elif test_model == 1:
        neural_network_test()
