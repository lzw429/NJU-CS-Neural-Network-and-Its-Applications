import argparse

import numpy as np

from project_1.activate_func import ReLU
from project_1.dataset import Dataset, DataLoader
from project_1.loss_func import MSELoss


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
    def __init__(self, act_func=ReLU(), loss_func=MSELoss()):
        # functions
        self.act = act_func.activate
        self.act_grad = act_func.grad
        self.loss = loss_func.loss
        self.loss_grad = loss_func.grad

        # parameters
        self.w_in = np.random.randn(n_h, n_in) * 0.01
        self.b_in = np.zeros(shape=(n_h))

        self.w_h = []  # (n_h, n_h)
        self.b_h = []  # (n_h)
        for layer_idx in range(n_l):
            self.w_h.append(np.random.randn(n_h, n_h) * 0.01)
            self.b_h.append(np.zeros(shape=(n_h)))

        self.w_out = np.random.randn(1, n_h) * 0.01
        self.b_out = np.zeros(shape=(1))

        self.Z_h
        self.A_h

    def forward(self, X):
        """
        The forward propagation of this NN
        :param X: the input tensor [n_batch, n_in]
        :return: the model output [n_batch, n_out]
        """
        self.X = np.array(X)
        self.Z_in = np.matmul(self.X, self.w_in.T) + self.b_in  # (n_batch, n_h)
        self.A_in = self.act(self.Z_in)  # (n_batch, n_h)
        self.A = self.A_in

        for l in range(n_l):  # for each layer
            self.Z = np.matmul(self.A, self.w_h[l]) + self.b_h[l]  # (n_batch, n_h)
            self.Z_h.append(self.Z)
            self.A = self.act(self.Z)  # (n_batch, n_h)
            self.A_h.append(self.A)
        self.Z = np.matmul(self.A, self.w_out.T) + self.b_out  # (n_batch, 1)
        self.A = np.squeeze(self.act(self.Z))
        self.w_h.append(self.w_out)
        self.Z_h.append(self.Z)
        self.A_h.append(self.A)
        return self.A  # (n_batch)

    def loss(self, y_pred, Y):
        return self.loss(y_pred, Y)

    def back_prob(self, Y):
        self.dw_out = np.mean(self.loss_grad(self.A, Y) * self.act_grad(self.Z) * self.A_h[n_l - 1], axis=0)  # (n_h)
        self.db_out = np.mean(self.loss_grad(self.A, Y) * self.act_grad(self.Z), axis=0)  # (1)
        self.dw_h = [np.zeros([n_batch, n_h])] * (n_l)
        self.db_h = [np.zeros([n_batch])] * (n_l)
        self.dw_h.append(self.dw_out)
        for l in range(n_l - 1, -1, -1):  # for each layer, l belongs to [n_l - 1, 0]
            if l != 0:
                x = self.A_h[l - 1]  # (n_batch, n_l)
            else:
                x = self.A_in  # (n_batch, n_l)
            self.dw_h[l] = - np.dot(
                self.act_grad(self.Z_h[l]).T * np.sum(np.dot(self.dw_h[l + 1], np.squeeze(self.w_h[l + 1]))),
                x)  # (n_h, n_h)
            self.db_h[l] = - np.mean(self.act_grad(self.Z_h[l]).T * np.sum(
                np.dot(self.dw_h[l + 1], np.squeeze(self.w_h[l + 1]))))

        self.dw_in = - np.dot(self.act_grad(self.Z_in).T * np.sum(np.dot(self.dw_h[0], self.w_h[0].T)),
                              self.X)  # (n_h , n_in)
        self.db_in = - np.mean(self.act_grad(self.Z_in).T * np.sum(np.dot(self.dw_h[0], self.w_h[0].T)))

    def update_parameters(self):
        self.w_out -= lr * self.dw_out
        self.b_out -= lr * self.db_out
        for l in range(n_l - 1, -1, -1):  # for each layer
            self.w_h[l] -= lr * self.dw_h[l]
            self.b_h[l] -= lr * self.db_h[l]
        self.w_in -= lr * self.dw_in
        self.b_in -= lr * self.db_in  # (n_h)

    def predict(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size", type=int, default=2)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_hidden_layer", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--num_iterate", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--shuffle", type=bool, default=True)
    args = parser.parse_args()

    n_h = args.hidden_size
    n_l = args.num_hidden_layer
    n_in = args.input_size
    n_batch = args.batch_size
    lr = args.lr

    X, Y = sample_generate()  # the inputs and golden results
    dataset = Dataset(X, Y)
    dataloader = DataLoader(dataset, 64, shuffle=args.shuffle)

    model = MultiLayerPerceptron()
    for epoch_idx in range(args.num_iterate):  # for each epoch
        for batch_idx in range(n_batch):  # for each batch
            inputs, golden = dataloader.get_batch(batch_idx)  # inputs and golden outputs for this batch
            y_pred = model.forward(inputs)  # predict
            local_loss = model.loss(y_pred, golden)  # get the loss
            model.back_prob(golden)  # back propagation
            model.update_parameters()  # update the parameters

            print("[INFO] epoch " + str(epoch_idx) + ", batch " + str(batch_idx) + ": " + str(local_loss))
