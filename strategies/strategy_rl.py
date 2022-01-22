"""
Playing a game and learning.
"""
import random
import numpy as np

import torch

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


class StrategyRL:

    def __init__(self, n_cards, epsilon=0.2):
        self.n_cards = n_cards
        self.nn = NeuralNetwork(n_cards)
        self.epsilon = epsilon

        self.dataset = {'piles': [], 'hands': [], 'n_cards': [], 'coup': []}

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
                print(chosen_coup)

            else:

                # Best NN choice
                choice_stat = self.nn((
                    torch.FloatTensor([nn_piles]), torch.FloatTensor([nn_hand]), torch.FloatTensor([nn_n_cards])))
                choice_stat = choice_stat.detach().numpy().ravel()
                choice_stat = [c if n in coups.keys() else 0. for n, c in zip(OUTPUT_ORDER, choice_stat)]
                chosen_coup = OUTPUT_ORDER[np.argmax(choice_stat)]

            # Save information
            self.dataset['piles'].append(nn_piles)
            self.dataset['hands'].append(nn_hand)
            self.dataset['n_cards'].append(nn_n_cards)
            self.dataset['coup'].append(chosen_coup)

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
