import numpy as np
import argparse

from project_1.dataset import Dataset, DataLoader
from project_1.activate_func import ReLU
from project_1.loss_func import Log_cosh


def fitted_func(x1, x2):
    return np.sin(x1) - np.cos(x2)


def sample_generate():
    x1 = np.arange(-5, 5, 0.01)
    x2 = np.arange(-5, 5, 0.01)
    sample_inputs = []
    sample_golden = []
    for i in x1:
        for j in x2:
            sample_inputs.append((i, j))  # the inputs: 2-d
            sample_golden.append(fitted_func(i, j))  # the golden outputs: 1-d
    return sample_inputs, sample_golden


class MultiLayerPerceptron:
    def __init__(self):
        self.w_in = np.random.randn(n_h, n_in) * 0.01
        self.b_in = np.zeros(shape=(n_h))

        self.w_h = []
        self.b_h = []
        for layer_idx in range(n_l):
            self.w_h.append(np.random.randn(n_h, n_h) * 0.01)
            self.b_h.append(np.zeros(shape=(n_h)))

        self.w_out = np.random.randn(1, n_h) * 0.01
        self.b_out = np.zeros(shape=(1))

    def forward(self, X):
        """
        The forward propagation of this NN
        :param X: the input tensor [n_batch, n_in]
        :return: the model output [n_batch, n_out]
        """

        self.Z_in = np.matmul(np.array(X), self.w_in.T) + self.b_in  # (n_batch, n_h)
        self.A_in = ReLU(self.Z_in)  # (n_batch, n_h)
        self.A = self.A_in

        self.Z_h = []
        for layer_idx in range(n_l):  # for each layer
            self.Z = np.matmul(self.A, self.w_h[layer_idx]) + self.b_h[layer_idx]  # (n_batch, n_h)
            self.Z_h.append(self.Z)
            self.A = ReLU(self.Z)  # (n_batch, n_h)
        self.Z = np.matmul(self.A, self.w_out.T) + self.b_out  # (n_batch, 1)
        self.A = ReLU(self.Z)
        return np.squeeze(self.A)  # (n_batch)

    def loss(self, y_pred, Y):
        return Log_cosh(y_pred, Y)

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
    n_batch = args.batch_size

    X, Y = sample_generate()  # the inputs and golden results
    dataset = Dataset(X, Y)
    dataloader = DataLoader(dataset, 64)

    model = MultiLayerPerceptron()
    for epoch_idx in range(args.num_iterate):  # for each epoch
        for batch_idx in range(n_batch):  # for each batch
            inputs, golden = dataloader.get_batch(batch_idx)  # inputs and golden outputs for this batch
            y_pred = model.forward(inputs)
            local_loss = model.loss(y_pred, golden)
