"""
Playing a game and learning.
"""
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn.functional as F

from reinforcement_learning.nn import NeuralNetwork

OUTPUT_ORDER = ['up_1', 'down_1', 'up_2', 'down_2', 'up_3', 'down_3', 'up_4', 'down_4', 'stop']


def get_possible_coups(hand, piles, cards_to_play):
    """
    Return all the possible coups from the current state.

    Args:
        hand (list of int): List of cards in the hand.
        piles (dict): List of cards on the piles.
        cards_to_play (int): Number of cards that need to be played.

    Returns (dict): The list of action that may be chosen.

    """

    actions = {}
    if cards_to_play == 0:
        actions['stop'] = True

    for pile_id in [1, 2]:
        scores = hand - piles[pile_id][-1]
        scores[scores <= 0] = 100
        if np.min(scores) != 100:
            actions['up_' + str(pile_id)] = (hand[np.argmin(scores)], pile_id)

        scores = hand - piles[pile_id][-1]
        jocker = hand[scores == -10]
        if len(jocker) == 1:
            actions['down_' + str(pile_id)] = (jocker[0], pile_id)

    for pile_id in [3, 4]:
        scores = piles[pile_id][-1] - hand
        scores[scores <= 0] = 100
        if np.min(scores) != 100:
            actions['down_' + str(pile_id)] = (hand[np.argmin(scores)], pile_id)

        scores = piles[pile_id][-1] - hand
        jocker = hand[scores == -10]
        if len(jocker) == 1:
            actions['up_' + str(pile_id)] = (jocker[0], pile_id)

    return actions


class TheGameDataset(Dataset):

    def __init__(self, nn_data):
        self.nn_data = nn_data

    def __len__(self):
        return len(self.nn_data[list(self.nn_data.keys())[0]])

    def __getitem__(self, idx):
        piles = torch.FloatTensor(self.nn_data['piles'][idx])
        hands = torch.FloatTensor(self.nn_data['hands'][idx])
        n_cards = torch.FloatTensor(self.nn_data['n_cards'][idx])
        coup = torch.FloatTensor(self.nn_data['coup'][idx])
        return piles, hands, n_cards, coup


class StrategyRL:

    def __init__(self, n_cards, epsilon=0.3):
        self.n_cards = n_cards
        self.nn = NeuralNetwork(n_cards)
        self.nn.load_last()
        self.epsilon = epsilon
        self.score = np.nan

        self.nn_data = {'piles': [], 'hands': [], 'n_cards': [], 'coup': []}

    def chose_coups(self, player, table):
        hand = np.array(player.hand.copy())
        piles = {p: table.piles[p].copy() for p in range(1, 5)}
        n_to_play = table.cards_to_play()
        end_game = False

        out_cards = []
        out_piles = []
        for i in range(len(hand)):
            # print('hand', hand)
            # print('piles', piles)
            # print('cards_to_play', n_to_play)
            coups = get_possible_coups(hand, piles, n_to_play)

            # If no coup is possible
            if len(coups.keys()) == 0:
                end_game = True
                break

            # Create nn data
            nn_piles = [piles[k][-1] / 98 for k in piles]

            nn_hand = np.zeros((2, self.n_cards))
            for i in range(self.n_cards):
                if i < len(hand):
                    nn_hand[0, i] = hand[i] / 98
                else:
                    nn_hand[0, i] = np.random.uniform()
                    nn_hand[1, i] = 1

            nn_n_cards = [n_to_play]

            # Choice
            if np.random.uniform() < self.epsilon:

                # Random coup chosen
                chosen_coup = random.choice(list(coups.keys()))

            else:

                # Best NN choice
                choice_stat = self.nn((
                    torch.FloatTensor(np.array([nn_piles])),
                    torch.FloatTensor(np.array([nn_hand])),
                    torch.FloatTensor(np.array([nn_n_cards]))))
                choice_stat = choice_stat.detach().numpy().ravel()
                choice_stat = [c if n in coups.keys() else 0. for n, c in zip(OUTPUT_ORDER, choice_stat)]
                chosen_coup = OUTPUT_ORDER[np.argmax(choice_stat)]

            # Save information
            self.nn_data['piles'].append(nn_piles)
            self.nn_data['hands'].append(nn_hand)
            self.nn_data['n_cards'].append(nn_n_cards)
            self.nn_data['coup'].append([coup == chosen_coup for coup in OUTPUT_ORDER])

            # If the coup is to stop
            if chosen_coup == 'stop':
                break

            # Add one card to the output
            c, p = coups[chosen_coup]
            out_cards.append(c)
            out_piles.append(p)

            # Update the game
            n_to_play = np.max([n_to_play - 1, 0])
            hand = np.setdiff1d(hand, [c])
            piles[p].append(c)
            # print(c, p)
            # print()

        return out_piles, out_cards, end_game

    def loss(self, choice_stat, coup):
        return F.cross_entropy(choice_stat, coup) + self.score

    def train_nn(self, score, n_epoch=50):
        self.score = score

        dataset = TheGameDataset(self.nn_data)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = optim.Adam(self.nn.parameters(), lr=1e-2)

        loss_epoch = []
        for epoch in range(n_epoch):

            loss_train = []
            for nn_piles, nn_hand, nn_n_cards, coup in dataloader:

                optimizer.zero_grad()
                choice_stat = self.nn((nn_piles, nn_hand, nn_n_cards))
                loss_train_subset = self.loss(choice_stat, coup)
                loss_train_subset.backward()
                optimizer.step()

                loss_train.append(loss_train_subset)

            loss = torch.sum(torch.stack(loss_train))

            print('epoch %d - loss train: %.6f' % (epoch, loss))
            loss_epoch.append(loss)
            epoch += 1

        self.nn.save()

        return loss_epoch
