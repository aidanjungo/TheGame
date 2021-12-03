# Not well implemented, but this is a good strategy anyway.

def caluculate_diffs(table, player):

    diffs = {1: [], 2: [], 3: [], 4: []}

    for n, card_in_pile in table.piles.items():
        for card in player.hand:
            if n < 3:
                diffs[n].append(card - card_in_pile[-1])
            else:
                diffs[n].append(-card + card_in_pile[-1])

    return diffs


def strategy_simple(player, table):
    """This stratery just play the minimum number of card.
    If possible card(s) with -10/10 and complete with card(s) with the minimum difference
    with the piles.

    """

    cards = []
    piles = []

    diffs = caluculate_diffs(table, player)

    cards_played = 0

    for n, diff in diffs.items():

        if -10 in diff:
            cards.append(player.hand[diff.index(-10)])
            diffs[n][diff.index(-10)] = 999
            piles.append(n)
            cards_played += 1

        for i, dif in enumerate(diff):
            if dif < 0:
                diffs[n][i] = 999

    while cards_played < table.cards_to_play():

        remaining_cards = 0
        for values in diffs.values():
            for v in values:
                if v != 999:
                    remaining_cards += 1
        if remaining_cards + cards_played < table.cards_to_play():
            return [], [], True

        min_diff = min(diffs[1] + diffs[2] + diffs[3] + diffs[4])

        if diffs[1] == diffs[2] and min_diff in diffs[1]:
            cards.append(player.hand[diffs[1].index(min_diff)])
            piles.append(1)
            diffs[1][diffs[1].index(min_diff)] = 999
            diffs[2][diffs[2].index(min_diff)] = 999
            cards_played += 1

        elif diffs[3] == diffs[4] and min_diff in diffs[3]:
            cards.append(player.hand[diffs[3].index(min_diff)])
            piles.append(3)
            diffs[3][diffs[3].index(min_diff)] = 999
            diffs[4][diffs[4].index(min_diff)] = 999
            cards_played += 1

        else:
            for n, diff in diffs.items():
                if min_diff in diff:
                    cards.append(player.hand[diff.index(min_diff)])
                    piles.append(n)
                    diffs[n][diff.index(min_diff)] = 999
                    cards_played += 1
                    break

    return piles, cards, False
