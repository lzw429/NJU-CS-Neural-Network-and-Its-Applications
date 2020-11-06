import numpy as np
import argparse

from project_1.func import ReLU


def fitted_func(x1, x2):
    return np.sin(x1) - np.cos(x2)


def sample_generate():
    x1 = np.arange(-5, 5, 0.01)
    x2 = np.arange(-5, 5, 0.01)
    sample_inputs = []
    sample_golden = []
    for i in x1:
        for j in x2:
            sample_inputs.append((i, j))
            sample_golden.append(fitted_func(i, j))
    return sample_inputs, sample_golden


class MultiLayerPerceptron:
    def __init__(self):
        self.w_in = np.random.randn(n_h, n_in) * 0.01
        self.b_in = np.zeros(shape=(n_h, 1))

        self.w_h = []
        self.b_h = []
        for layer_idx in range(n_l):
            self.w_h.append(np.random.randn(n_h, n_h) * 0.01)
            self.b_h.append(np.zeros(shape=(n_h, 1)))

        self.w_out = np.random.randn(1, n_h) * 0.01
        self.b_out = np.zeros(shape=(1, 1))

    def forward(self, X):
        self.Z_in = np.dot(self.w_in, X) + self.b_in
        self.A_in = ReLU(self.Z_in)
        self.A = self.A_in

        self.Z_h = []
        for layer_idx in range(n_l):
            self.Z = np.dot(self.w_h[layer_idx], self.A) + self.b_h[layer_idx]
            self.Z_h.append(self.Z)
            self.A = ReLU(self.Z)
        self.Z = np.dot(self.w_out, self.A) + self.b_out
        self.A = ReLU(self.Z)

    def loss(self, y_pred, Y):

        pass

    def back_prob(self):
        pass

    def update_parameters(self):
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size", type=int, default=2)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_hidden_layer", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=10)
    parser.add_argument("--num_iterate", type=int, default=10000)
    args = parser.parse_args()

    n_h = args.hidden_size
    n_l = args.num_hidden_layer
    n_in = args.input_size

    X, Y = sample_generate()
    model = MultiLayerPerceptron()
    for epoch_idx in range(args.num_iterate):
        for batch_idx in range(args.batch_size):
            y_pred = model.forward(X)
            print(y_pred)
