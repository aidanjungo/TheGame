"""
Neural Network to play the game.
"""
import os
import re
import numpy as np

import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class NeuralNetwork(nn.Module):
    def __init__(self, n_cards, hidden_layers=(50, 100, 50)):
        """
        Instanciate the neural network

        Args:
            n_cards (in): number of cards in players hand
            hidden_layers (tuple of int): number of neuron in hidden layers
        """
        super(NeuralNetwork, self).__init__()
        self.n_cards = n_cards
        self.conv = nn.Conv1d(2, 1, 1)

        self.sequential = nn.Sequential()
        self.id_model = 0
        layers = []
        layers.append(nn.Linear(4 + n_cards + 1, hidden_layers[0]))

        for i, input_len in enumerate(hidden_layers):
            if i + 1 < len(hidden_layers):
                output_len = hidden_layers[i + 1]
            else:
                output_len = 9

            layers.append(nn.Sigmoid())
            layers.append(nn.Linear(input_len, output_len))

        layers.append(nn.Softmax(1))

        self.sequential = nn.Sequential(*layers)

    def forward(self, states):
        piles, hands, n_cards = states
        n_data = piles.size(0)

        # Convolution for the cards
        hands_conv = self.conv(hands).resize(n_data, self.n_cards)

        # Sequential for all the data
        x = torch.cat((piles, hands_conv, n_cards), 1)
        return self.sequential(x)

    def save(self):
        os.makedirs('saved_models', exist_ok=True)
        torch.save(
            self.sequential,
            os.path.join('saved_models', str(self.n_cards) + '_' + str(self.id_model) + '.pth'))

    def load(self, id_model=0):
        self.sequential = torch.load(
            os.path.join('saved_models', str(self.n_cards) + '_' + str(id_model) + '.pth'))
        self.sequential.eval()
        self.id_model = id_model + 1

    def load_last(self):
        if 'saved_models' in os.listdir(os.curdir):
            list_files = os.listdir('saved_models')
            list_ids = [int(f[2:-4]) for f in list_files if re.match(str(self.n_cards) + '_[0-9]+.pth', f) is not None]
            if len(list_ids) > 0:
                id_last_model = np.max(list_ids)
                self.load(id_model=id_last_model)
                for id in list_ids:
                    if id_last_model - 10 > id:
                        os.remove(os.path.join('saved_models', str(self.n_cards) + '_' + str(id) + '.pth'))


if __name__ == "__main__":
    n_cards = 5
    nn = NeuralNetwork(n_cards)
    print(nn)

    # states = torch.FloatTensor(10 * [
    #     [1, 1, 100, 100] + [i for i in range(n_cards)] + [0 for i in range(n_cards)] + [2]])
    # states = states.resize(len(states), 1)
    states = (
        torch.FloatTensor(10 * [[1, 1, 100, 100]]),
        torch.FloatTensor(10 * [[[i for i in range(n_cards)], [0 for i in range(n_cards)]]]),
        torch.FloatTensor(10 * [[2]]),
    )
    print(states)

    print(nn(states))

    nn.save()
    nn.load(id_model=0)
    nn.load_last()