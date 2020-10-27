import torch
import torch.nn as nn


class Neuron(nn.Module):

    def __init__(self):
        self.neuron = nn.Linear(4, 1)

    def forward(self, x):
        return self.neuron(x)


if __name__ == '__main__':
    neuron = Neuron()
    input = torch.tensor([[1., -1.], 
                          [1., -1.]])

