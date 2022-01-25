# Possible improvements:
# - Improve code to run faster

import numpy as np


def calculate_diffs(piles, hand):
    """Calculate the diffs between the cards in hand and the piles.
    Return a list of tuples (diff, pile, card) ordered by diff.
    """

    diffs = {}
    cards = {}
    diffs_list = [] * len(hand) * len(piles)

    for n, pile in piles.items():
        sign = np.sign(n - 2.5)
        cards[n] = np.array(hand)
        diffs[n] = -sign * cards[n] + sign * pile
        diffs[n][diffs[n] <= 0] = np.nan

        for d in diffs[n]:
            if not np.isnan(d):
                diffs_list.append((d, n, cards[n][np.where(diffs[n] == d)][0]))

    diffs_list.sort(key=lambda x: x[0])
    diffs_list_new = []

    copy_hand = set(hand.copy())
    for d, p, card in diffs_list:

        if card in copy_hand:

            copy_hand.remove(card)
            diffs_list_new.append((int(d), p, card))

    return diffs_list_new


def strategy_n_diff(player, table, acceptable_diff):
    """This strategy try to:
    - first to play the card with diff of 10.
    Then loop until min number of cards is reach and not bellow acceptable_diff
    - play card with the minimum diff until the number of card to play is reached
    - play additional card if the diff is less than the acceptable_diff
    - check if the card with diff of 10 is playable
    """

    hand = player.hand.copy()

    piles = {p: table.piles[p][-1] for p in range(1, 5)}

    piles_to_play = []
    cards_to_play = []

    end_game = False

    for p in piles.keys():
        for c in hand:
            card = piles[p] + np.sign(p - 2.5) * 10
            if card in hand:
                cards_to_play.append(int(card))
                piles_to_play.append(p)
                hand.remove(card)
                piles[p] = card
            else:
                break

    if table.cards_to_play() == 1:
        acceptable_diff = 1

    playing = True
    while playing:

        diffs = calculate_diffs(piles, hand)

        if not diffs:
            break

        if len(cards_to_play) >= table.cards_to_play():
            if diffs[0][0] > acceptable_diff:
                playing = False

        if playing:
            piles_to_play.append(diffs[0][1])
            cards_to_play.append(diffs[0][2])
            hand.remove(diffs[0][2])
            piles[diffs[0][1]] = diffs[0][2]

        for p in range(1, 5):
            for c in hand:
                card = piles[p] + np.sign(p - 2.5) * 10
                if card in hand:
                    cards_to_play.append(int(card))
                    piles_to_play.append(p)
                    hand.remove(card)
                    piles[p] = card
                else:
                    break

    if len(cards_to_play) < table.cards_to_play():
        end_game = True

    return piles_to_play, cards_to_play, end_game
