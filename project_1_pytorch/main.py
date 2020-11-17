import argparse

import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from project_1.main import sample_generate, fitted_func


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_in = nn.Linear(2, n_h)
        self.layer_h = []
        for l in range(n_l):
            self.layer_h.append(nn.Linear(n_h, n_h))
        self.layer_out = nn.Linear(n_h, 1)

    def forward(self, x, act_hidden=nn.Sigmoid(), act_out=nn.Sigmoid()):
        A = act_hidden(self.layer_in(x))
        for l in range(n_l):
            A = act_hidden(self.layer_h[l](A))
        A = act_out(self.layer_out(A))
        return A


class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, Y):
        self.inputs = torch.tensor(X, dtype=torch.float)
        self.golden = torch.tensor(Y, dtype=torch.float)

    def __getitem__(self, index):
        return self.inputs[index], self.golden[index]

    def __len__(self):
        return len(self.inputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size", type=int, default=2)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_hidden_layer", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=40)
    parser.add_argument("--num_epoch", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--shuffle", type=bool, default=True)
    args = parser.parse_args()

    n_h: int = args.hidden_size
    n_l: int = args.num_hidden_layer
    n_in: int = args.input_size
    batch_size: int = args.batch_size
    lr: float = args.lr

    model = Model()
    X, Y = sample_generate()  # the inputs and golden results
    dataset = Dataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # training
    train_finished = False
    for epoch_idx in range(args.num_epoch):
        running_loss = 0.0
        if train_finished:
            break
        for batch_idx, sample_batched in enumerate(dataloader):
            inputs, golden = sample_batched

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, golden)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch_idx + 1, batch_idx + 1, running_loss))
            if running_loss < 0.000001:
                train_finished = True
                break
            running_loss = 0.0
    print('[INFO] Finished Training')

    # validation
