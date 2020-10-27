import torch
import torch.nn as nn
import torch.optim as optim


class Neuron(nn.Module):

    def __init__(self):
        self.neuron = nn.Linear(4, 1)

    def forward(self, x):
        return self.neuron(x)


if __name__ == '__main__':
    neuron = Neuron()
    inputs = torch.tensor([[55.22, 56.34, 55.52, 55.53],
                           [56.34, 55.52, 55.53, 56.94],
                           [55.52, 55.53, 56.94, 58.88],
                           [55.53, 56.94, 58.88, 58.18],
                           [56.94, 58.88, 58.18, 57.09],
                           [58.88, 58.18, 57.09, 58.38],
                           [58.18, 57.09, 58.38, 58.54],
                           [57.09, 58.38, 58.54, 57.72],
                           [58.38, 58.54, 57.72, 58.02],
                           [58.54, 57.72, 58.02, 57.81],
                           [57.72, 58.02, 57.81, 58.71],
                           [58.02, 57.81, 58.71, 60.84],
                           [57.81, 58.71, 60.84, 61.08],
                           [58.71, 60.84, 61.08, 61.74],
                           [60.84, 61.08, 61.74, 62.16],
                           [61.08, 61.74, 62.16, 60.80]])

    golden = torch.tensor(
        [56.94, 58.88, 58.18, 57.09, 58.38, 58.54, 57.72, 58.02, 57.81, 58.71, 60.84, 61.08, 61.74, 62.16, 60.80,
         60.87])

    criterion = nn.MSELoss()
    neuron = Neuron()
    optimizer = optim.AdamW(neuron.parameters(), lr=0.0001)

    print("[INFO] Start Training")
    running_loss = 0.0
    for epoch_idx in range(5):
        optimizer.zero_grad()
        outputs = neuron(inputs)
        loss = criterion(outputs, golden)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print("[INFO] Finished Training")

    with torch.no_grad():
        outputs = neuron(inputs)
        print(outputs)
