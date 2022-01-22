"""
Neural Network to play the game.
"""
import os
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class NeuralNetwork(nn.Module):
    def __init__(self, n_cards, hidden_layers=(20, 20)):
        """
        Instanciate the neural network

        Args:
            n_cards (in): number of cards in players hand
            hidden_layers (tuple of int): number of neuron in hidden layers
        """
        super(NeuralNetwork, self).__init__()
        self.sequential = nn.Sequential()
        self.id_model = 0
        layers = []
        layers.append(nn.Linear(4 + n_cards, hidden_layers[0]))

        for i, input_len in enumerate(hidden_layers):
            if i + 1 < len(hidden_layers):
                output_len = hidden_layers[i + 1]
            else:
                output_len = 2 * n_cards

            layers.append(nn.Sigmoid())
            layers.append(nn.Linear(input_len, output_len))

        layers.append(nn.Softmax())

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)

    def save(self):
        os.makedirs('saved_models', exist_ok=True)
        torch.save(self.sequential, 'saved_models/' + str(self.id_model) + '.pth')

    def load(self, id_model=0):
        self.sequential = torch.load('saved_models/' + str(id_model) + '.pth')
        self.sequential.eval()
        self.id_model = id_model


if __name__ == "__main__":
    n_cards = 5
    nn = NeuralNetwork(n_cards)
    print(nn)

    states = torch.FloatTensor(10 * [[1, 1, 100, 100] + [i for i in range(n_cards)]])
    # states = states.resize(len(states), 1)
    print(states)

    print(nn(states))

    nn.save()
    nn.load(id_model=1)