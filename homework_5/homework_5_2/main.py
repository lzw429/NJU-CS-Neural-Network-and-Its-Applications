import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from homework_5_2.dataset import Dataset


class Model(nn.Module):
    def __init__(self, act_h=nn.LeakyReLU(), act_o=nn.Sigmoid()):
        super().__init__()
        self.layer_in = nn.Linear(64, hidden_size)
        self.layer_h = nn.Linear(hidden_size, hidden_size)
        self.layer_out = nn.Linear(hidden_size, 10)
        self.act_h = act_h
        self.act_o = act_o

    def forward(self, x):
        a = self.act_h(self.layer_in(x))
        a = self.act_h(self.layer_h(a))
        return self.act_o(self.layer_out(a))


def evaluate_model():
    global loss, batch_idx, sample_batched, inputs, golden, outputs
    loss = 0.0
    outputs_list = torch.tensor([], dtype=torch.float)
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(testing_dataloader):
            inputs, golden = sample_batched
            outputs = model(inputs)
            outputs_list = torch.cat((outputs_list, outputs), 0)
            loss += criterion(outputs, golden)
    print('validation loss: ' + str(loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--num_epoch', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--dir', type=str,
                        default='/Users/shuyiheng/Documents/GitHub/NJU-CS-Neural-Network-and-Its-Applications/homework_5/homework_5_2')
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_valid', type=bool, default=True)
    args = parser.parse_args()

    hidden_size = args.hidden_size
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    lr = args.lr

    training_dataset = Dataset(np.loadtxt(
        '/Users/shuyiheng/Documents/GitHub/NJU-CS-Neural-Network-and-Its-Applications/homework_5/homework_5_2/optdigits_data/optdigits.tra',
        delimiter=','))
    testing_dataset = Dataset(np.loadtxt(
        '/Users/shuyiheng/Documents/GitHub/NJU-CS-Neural-Network-and-Its-Applications/homework_5/homework_5_2/optdigits_data/optdigits.tes',
        delimiter=','))

    model = Model()
    optimizer = optim.AdamW(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

    if args.do_train:
        last_loss = 0.0
        for epoch_idx in range(num_epoch):
            running_loss = 0.0
            for batch_idx, sample_batched in enumerate(training_dataloader):
                inputs, golden = sample_batched

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, golden)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print('[%d] loss: %.3f' % (epoch_idx + 1, running_loss))
            if running_loss > last_loss:
                print('[WARN] the loss is increasing')
            last_loss = running_loss
            if running_loss < 0.0001:
                break
            if epoch_idx % 10 == 0:
                evaluate_model()

        print('[INFO] Finished Training')

    if args.do_valid:
        evaluate_model()
