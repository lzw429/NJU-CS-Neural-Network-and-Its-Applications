import homework_5.homework_5_1
from homework_5_1.dataset import sample_generate, Dataset
from homework_5_1.neural_network import NeuralNetwork
from homework_5_1.perceptron import Perceptron
import torch.optim as optim


def perceptron_test():
    dataloader =
    optimizer = optim.AdamW(perceptron.parameters(), lr)
    criterion =

    last_loss = 0.0

def neural_network_test():

    last_loss = 0.0


if __name__ == '__main__':
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()
    lr = args.lr

    perceptron = Perceptron()
    neural_network = NeuralNetwork(n_h=2, hidden_size=20)
    X, Y = sample_generate()

    dataset = Dataset(X, Y)

