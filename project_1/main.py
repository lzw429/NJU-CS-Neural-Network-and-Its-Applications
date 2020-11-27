import argparse
import copy
import numpy as np

from project_1.activate_func import ReLU, Tanh, LeakyReLU, Sigmoid, Linear
from project_1.dataset import Dataset, DataLoader
from project_1.loss_func import MSELoss, LogCoshLoss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from project_1.optim import SGD, Adam, Optimizer, AdamW


def fitted_func(x1, x2):
    return np.sin(x1) - np.cos(x2)


def sample_generate():
    x1 = np.arange(-5, 5, 0.1)
    x2 = np.arange(-5, 5, 0.1)
    sample_inputs = []
    sample_golden = []
    for i in x1:
        for j in x2:
            sample_inputs.append((i, j))  # the inputs: 2-d
            sample_golden.append(fitted_func(i, j))  # the golden outputs: 1-d
    return np.array(sample_inputs), np.array(sample_golden)


class MultiLayerPerceptron:
    def __init__(self, optim: Optimizer = SGD(0.00005), hidden_act_func=LeakyReLU(), output_act_func=Linear(1),
                 loss_func=MSELoss()):
        # functions
        self.h_act = hidden_act_func.activate
        self.h_act_grad = hidden_act_func.grad
        self.o_act = output_act_func.activate
        self.o_act_grad = output_act_func.grad
        self.loss = loss_func.loss
        self.loss_grad = loss_func.grad

        # parameters
        self.w_in = np.random.randn(n_h, n_in) * 0.03
        self.b_in = np.zeros(shape=n_h)

        self.w_h = []  # (n_h, n_h)
        self.b_h = []  # (n_h)
        for l in range(n_l):
            self.w_h.append(np.random.randn(n_h, n_h) * 0.03)
            self.b_h.append(np.zeros(shape=n_h))

        self.w_out = np.random.randn(1, n_h) * 0.03
        self.b_out = np.zeros(shape=1)
        self.w_h.append(self.w_out)
        self.b_h.append(self.b_out)

        self.Z_h = list(range(n_l + 1))
        self.A_h = list(range(n_l + 1))

        self.optim = {'w_in': copy.deepcopy(optim), 'b_in': copy.deepcopy(optim), 'w_out': copy.deepcopy(optim),
                      'b_out': copy.deepcopy(optim)}
        for l in range(n_l):
            self.optim['w_h_' + str(l)] = copy.deepcopy(optim)
            self.optim['b_h_' + str(l)] = copy.deepcopy(optim)

    def forward(self, X):
        """
        The forward propagation of this NN
        :param X: the input tensor [n_batch, n_in]
        :return: the model output [n_batch, n_out]
        """
        self.X = np.array(X)
        self.Z_in = np.matmul(self.X, self.w_in.T) + self.b_in  # (n_batch, n_h)
        self.A_in = self.h_act(self.Z_in)  # (n_batch, n_h)

        for l in range(n_l):  # for each layer
            self.Z_h[l] = np.dot(self.A_in, self.w_h[l]) + self.b_h[l]  # (n_batch, n_h)
            self.A_h[l] = self.h_act(self.Z_h[l])  # (n_batch, n_h)
        self.Z = np.matmul(self.A_h[n_l - 1], self.w_out.T) + self.b_out  # (n_batch, 1)
        self.A = np.squeeze(self.o_act(self.Z))
        self.w_h[n_l] = self.w_out
        self.Z_h[n_l] = self.Z
        self.A_h[n_l] = self.A
        return self.A  # (n_batch)

    def loss(self, y_pred, Y):
        return self.loss(y_pred, Y)

    def backward(self, Y):
        self.dw_out = np.mean(self.loss_grad(self.A, Y) * self.o_act_grad(self.Z) * self.A_h[n_l - 1], axis=0)  # (n_h)
        self.db_out = np.mean(self.loss_grad(self.A, Y) * self.o_act_grad(self.Z), axis=0)  # (1)
        self.dw_h = [np.zeros([batch_size, n_h])] * n_l
        self.db_h = [np.zeros([batch_size])] * n_l
        self.dw_h.append(self.dw_out)
        for l in range(n_l - 1, -1, -1):  # for each layer, l belongs to [n_l - 1, 0]
            if l != 0:
                x = self.A_h[l - 1]  # (n_batch, n_l)
            else:
                x = self.A_in  # (n_batch, n_l)
            self.dw_h[l] = - 1.0 / batch_size * np.dot(
                self.h_act_grad(self.Z_h[l]).T * np.sum(np.dot(self.dw_h[l + 1], np.squeeze(self.w_h[l + 1]))),
                x)  # (n_h, n_h)
            self.db_h[l] = - np.mean(self.h_act_grad(self.Z_h[l]).T * np.sum(
                np.dot(self.dw_h[l + 1], np.squeeze(self.w_h[l + 1]))), axis=1)

        self.dw_in = - 1.0 / batch_size * np.dot(
            self.h_act_grad(self.Z_in).T * np.sum(np.dot(self.dw_h[0], np.squeeze(self.w_h[0]))),
            self.X)  # (n_h , n_in)
        self.db_in = - np.mean(self.h_act_grad(self.Z_in).T * np.sum(np.dot(self.dw_h[0], np.squeeze(self.w_h[0]))),
                               axis=1)

    def update_parameters(self):
        self.optim['w_out'].update_parameters(self.w_out, self.dw_out)
        self.optim['b_out'].update_parameters(self.b_out, self.db_out)
        for l in range(n_l - 1, -1, -1):  # for each layer
            self.optim['w_h_' + str(l)].update_parameters(self.w_h[l], self.dw_h[l])
            self.optim['b_h_' + str(l)].update_parameters(self.b_h[l], self.db_h[l])
        self.optim['w_in'].update_parameters(self.w_in, self.dw_in)
        self.optim['b_in'].update_parameters(self.b_in, self.db_in)

    def predict(self):
        pass


def evaluate_model():
    eval_loss = 0.0
    outputs_list = np.array([])
    for batch_idx in range(dataloader.get_num_batch()):  # for each batch
        inputs, golden = dataloader.get_batch(batch_idx)  # inputs and golden outputs for this batch
        y_pred = model.forward(inputs)  # predict
        eval_loss += model.loss(y_pred, golden)  # get the loss
        outputs_list = np.append(outputs_list, y_pred)
    if args.plot:
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(X[:, 0], X[:, 1], outputs_list, color='r')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size", type=int, default=2)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--num_hidden_layer", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--num_epoch", type=int, default=500000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()

    n_h: int = args.hidden_size
    n_l: int = args.num_hidden_layer
    n_in: int = args.input_size
    batch_size: int = args.batch_size
    lr: float = args.lr
    log_file = open(args.dir + "/log.txt", mode="w")

    X, Y = sample_generate()  # the inputs and golden results
    dataset: Dataset = Dataset(X, Y)
    dataloader: DataLoader = DataLoader(dataset, batch_size, shuffle=args.shuffle)

    if args.plot:
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(X[:, 0], X[:, 1], Y)
        plt.show()

    model = MultiLayerPerceptron(AdamW(lr))
    last_loss = 0.0
    for epoch_idx in range(args.num_epoch):  # for each epoch
        running_loss = 0.0
        for batch_idx in range(dataloader.get_num_batch()):  # for each batch
            inputs, golden = dataloader.get_batch(batch_idx)  # inputs and golden outputs for this batch
            y_pred = model.forward(inputs)  # predict
            running_loss += model.loss(y_pred, golden)  # get the loss
            model.backward(golden)  # back propagation
            model.update_parameters()  # update the parameters

        print('[%d] loss: %.3f' % (epoch_idx + 1, running_loss), file=log_file)  # print to file
        log_file.flush()
        print('[%d] loss: %.3f' % (epoch_idx + 1, running_loss))  # print to console
        if running_loss > last_loss != 0.0:
            print("[WARN] the loss is increasing")
        last_loss = running_loss
        if running_loss < 0.001:
            break
        if epoch_idx % 50 == 0:
            evaluate_model()
