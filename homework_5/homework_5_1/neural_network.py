import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, n_h, hidden_size, h_act=nn.LeakyReLU(), o_act=nn.Sigmoid()):
        """
        The initialization of the neural network
        :param n_h: the number of hidden layers
        :param hidden_size: the hidden size
        :param h_act: the activation function of the hidden layer
        :param o_act: the activation function of the output layer
        """
        super().__init__()
        self.n_h = n_h
        self.hidden_size = hidden_size
        self.h_act = h_act
        self.o_act = o_act

        self.layer_in = nn.Linear(2, self.hidden_size)
        self.layer_h = []
        for l in range(self.n_h):
            self.layer_h.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.layer_out = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        a = self.h_act(self.layer_in(x))
        for l in range(self.n_h):
            a = self.h_act(self.layer_h[l](a))
        return self.o_act(self.layer_out(a))
