import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, hidden_size=50, act_h=nn.LeakyReLU(), act_o=nn.Tanh(), n_l=5):
        super().__init__()
        self.n_l = n_l
        self.hidden_size = hidden_size
        self.act_h = act_h
        self.act_o = act_o

        self.layer_in = nn.Linear(1, self.hidden_size)
        self.layer_h = []
        for l in range(self.n_l):
            self.layer_h.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.layer_out = nn.Linear(self.hidden_size, 1)
        self.a_in = None
        self.a_h = [0] * self.n_l
        self.a_out = None

    def forward(self, x):
        a = self.act_h(self.layer_in(x))
        self.a_in = a
        for l in range(self.n_l):
            a = self.act_h(self.layer_h[l](a))
            self.a_h[l] = a
        self.a_out = self.act_o(self.layer_out(a))
        return self.a_out

    def predict(self, x, layer):
        """
        Get the output of the specified layer
        :param x: the input tensors
        :param layer: 0 is the input layer, n_l + 1 is the output layer, [1...n_l] are the hidden layers
        :return: the output of the specified layer of the network
        """
        if layer == 0:
            return self.a_in
        if layer > self.n_l:
            return self.a_out
        return self.a_h[layer]
