import torch
import torch.nn as nn
import torch.optim as optim


class Neuron(nn.Module):

    def __init__(self):
        super().__init__()
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
    optimizer = optim.AdamW(neuron.parameters(), lr=0.00005)

    print("[INFO] Start Training")
    for epoch_idx in range(310000):
        optimizer.zero_grad()
        outputs = torch.squeeze(neuron(inputs))
        loss = criterion(outputs, golden)
        loss.backward()
        optimizer.step()
        if epoch_idx % 10000 == 0:
            print("[INFO] epoch: " + str(epoch_idx) + ", running_loss: " + str(loss.item()))

    print("[INFO] Finished Training")

    with torch.no_grad():
        outputs = neuron(inputs)
        print(outputs)

    # the weight: [-0.2057, 0.5655, -0.5762, 1.0367]
    # the bias: [10.8877]
    # the loss: [0.6690]
    # prediction: [[56.9727], [57.7346], [59.1079], [58.0597], [58.1401], [59.3105], [58.2607], [58.2722], [58.8808], [57.9936], [59.3860], [60.8952], [60.4690], [62.0344], [61.7871], [60.4591]]
