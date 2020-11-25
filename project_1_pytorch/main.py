import argparse

import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from project_1.dataset import normalization
from project_1.main import sample_generate, fitted_func


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_in = nn.Linear(2, n_h)
        self.layer_h = []
        for l in range(n_l):
            self.layer_h.append(nn.Linear(n_h, n_h))
        self.layer_out = nn.Linear(n_h, 1)

    def forward(self, x, act_hidden=nn.LeakyReLU(), act_out=nn.LeakyReLU()):
        A = act_hidden(self.layer_in(x))
        for l in range(n_l):
            A = act_hidden(self.layer_h[l](A))
        A = act_out(self.layer_out(A))
        return A


class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, Y):
        X = normalization(X)
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
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_hidden_layer", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=150)
    parser.add_argument("--num_epoch", type=int, default=500000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--dir", type=str)
    parser.add_argument("--optimizer", type=str, default="adamw")  # 'sgd', 'adam', 'adamw'
    args = parser.parse_args()

    n_h: int = args.hidden_size
    n_l: int = args.num_hidden_layer
    n_in: int = args.input_size
    batch_size: int = args.batch_size
    lr: float = args.lr
    log_file = open(args.dir + "/log.txt", mode="w")

    model = Model()
    X, Y = sample_generate()  # the inputs and golden results
    # draw the fitted surface
    if args.plot:
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(X[:, 0], X[:, 1], Y)
        plt.show()

    dataset = Dataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    criterion = nn.MSELoss()

    # training
    last_loss = 0.0
    for epoch_idx in range(args.num_epoch):
        running_loss = 0.0
        for batch_idx, sample_batched in enumerate(dataloader):
            inputs, golden = sample_batched

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(torch.squeeze(outputs), torch.squeeze(golden))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print('[%d] loss: %.3f' % (epoch_idx + 1, running_loss), file=log_file)
        log_file.flush()
        print('[%d] loss: %.3f' % (epoch_idx + 1, running_loss))
        if running_loss > last_loss:
            print("[WARN] the loss is increasing")
        last_loss = running_loss
        if running_loss < 0.001:
            break

    print('[INFO] Finished Training')
    torch.save(model.state_dict(), args.dir + "model.pt")

    # validation
    # with torch.no_grad():
    #     for batch_idx, sample_batched in enumerate(dataloader):
    #         inputs, golden = sample_batched
    #         outputs = model(inputs)
    # outputs_list = torch.cat((outputs_list, outputs), 0)
    # plt.plot(x, outputs_list, color='y')
    # plt.show()
