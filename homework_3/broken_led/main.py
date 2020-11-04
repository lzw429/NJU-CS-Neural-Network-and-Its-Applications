import torch
import torch.nn as nn
import torch.optim as optim

from homework3.broken_led.data_gen import data_generation


class Neuron(nn.Module):

    def __init__(self):
        super().__init__()
        self.neuron = nn.Linear(7, 1)

    def forward(self, x):
        return torch.sigmoid(self.neuron(x))


def train(golden, model, optimizer):
    for epoch_idx in range(221000):
        optimizer.zero_grad()
        outputs = torch.squeeze(model(inputs))
        loss = criterion(outputs, golden)
        loss.backward()
        optimizer.step()
        if epoch_idx % 1000 == 0:
            print("[INFO] epoch: " + str(epoch_idx) + ", running_loss: " + str(loss.item()))


if __name__ == '__main__':
    neuron_2 = Neuron()
    data_inputs, data_outputs_2, data_outputs_3 = data_generation()
    inputs = torch.tensor(data_inputs, dtype=torch.float)

    golden_2 = torch.tensor(data_outputs_2, dtype=torch.float)
    golden_3 = torch.tensor(data_outputs_3, dtype=torch.float)

    criterion = nn.MSELoss()
    neuron_2 = Neuron()
    neuron_3 = Neuron()
    optimizer_2 = optim.AdamW(neuron_2.parameters(), lr=0.00005)
    optimizer_3 = optim.AdamW(neuron_3.parameters(), lr=0.00005)

    print("[INFO] Start Training P(2|x)")
    train(golden_2, neuron_2, optimizer_2)
    print("\n[INFO] Start Training P(3|x)")
    train(golden_3, neuron_3, optimizer_3)

    print("[INFO] Finished Training")

    with torch.no_grad():
        outputs_2 = neuron_2(inputs)
        print(outputs_2)
        outputs_3 = neuron_3(inputs)
        print(outputs_3)
