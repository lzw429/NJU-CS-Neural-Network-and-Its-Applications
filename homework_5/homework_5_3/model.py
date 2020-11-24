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

    def forward(self, x):
        a = self.act_h(self.layer_in(x))
        for l in range(self.n_l):
            a = self.act_h(self.layer_h[l](a))
        return self.act_o(self.layer_out(a))
