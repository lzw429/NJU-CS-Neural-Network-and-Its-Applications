import torch.nn as nn


class Perceptron(nn.Module):
    def __init__(self, perceptron=nn.Linear(2, 1), act=nn.Sigmoid()):
        """
        The initialization of the perceptron
        :param perceptron: the single layer
        :param act: the activation function
        """
        super().__init__()
        self.perceptron = perceptron
        self.act = act

    def forward(self, x):
        return self.act(self.perceptron(x))
