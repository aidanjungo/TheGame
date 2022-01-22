"""
Playing a game and learning.
"""
import random
import numpy as np


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

def strategy_rl(player, table, nn):
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

        if len(coups.keys()) == 0:
            end_game = True
            break

        # Random coup chosen
        chosen_coup = random.choice(list(coups.keys()))
        print(chosen_coup)

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

    print(out_cards, out_piles)

    return out_piles, out_cards, end_game
