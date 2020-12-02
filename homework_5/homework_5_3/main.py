import argparse

import numpy as np
import torch
import torch.utils.data
from torch import optim, nn
import matplotlib.pyplot as plt

from homework_5_3.dataset import sample_generate, Dataset
from homework_5_3.model import Model


def plot_prediction():
    with torch.no_grad():
        plt.plot(X, model(torch.tensor(X, dtype=torch.float).unsqueeze(1)), 'r')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=5)
    parser.add_argument('--num_epoch', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_layer', type=int, default=0)
    parser.add_argument('--dir', type=str,
                        default='/Users/shuyiheng/Documents/GitHub/NJU-CS-Neural-Network-and-Its-Applications/homework_5/homework_5_3')
    parser.add_argument('--do_train', type=bool, default=False)
    args = parser.parse_args()
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    lr = args.lr
    num_layer = args.num_layer

    X, Y = sample_generate()
    plt.plot(X, Y)
    plt.show()
    dataset = Dataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = Model(n_l=num_layer)
    optimizer = optim.AdamW(model.parameters(), lr)
    criterion = nn.MSELoss()

    if not args.do_train:
        model.load_state_dict(torch.load(args.dir + '/model.pt'))
    else:
        last_loss = 0.0
        for epoch_idx in range(num_epoch):
            running_loss = 0.0
            for batch_idx, sample_batched in enumerate(dataloader):
                inputs, golden = sample_batched

                optimizer.zero_grad()
                inputs = inputs.unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, golden)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print('[%d] loss: %.3f' % (epoch_idx + 1, running_loss))
            if running_loss > last_loss:
                print('[WARN] the loss is increasing')
            last_loss = running_loss
            if running_loss < 0.0005:
                break
            if epoch_idx % 100 == 0:
                plot_prediction()

        print('[INFO] Finished Training')
        torch.save(model.state_dict(), args.dir + '/model.pt')

    print('input layer weight: ' + str(model.layer_in.weight))
    print('input layer bias: ' + str(model.layer_in.bias))
    for l in range(num_layer):
        print('hidden layer ' + str(l) + ' weight: ' + str(model.layer_h[l].weight))
        print('hidden layer ' + str(l) + ' bias: ' + str(model.layer_h[l].bias))
    print('output layer weight: ' + str(model.layer_out.weight))
    print('output layer bias: ' + str(model.layer_out.bias))

    plot_prediction()
