import argparse

import torch
from torch import optim, nn

from homework_5_3.dataset import sample_generate, Dataset
from homework_5_3.model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=50)
    parser.add_argument("--num_epoch", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    lr = args.lr

    X, Y = sample_generate()
    dataset = Dataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = Model()
    optimizer = optim.AdamW(model.parameters(), lr)
    criterion = nn.MSELoss()

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
        if running_loss < 0.0001:
            break

    print('[INFO] Finished Training')
