import sys
import pytest

sys.path.append("..")

from thegame import Table, Player
from strategies.strategy_n_diff import strategy_n_diff, calculate_diffs


class TestStrategyNDiff:
    """
    TestCase of the strategy 'strategy_n_diff'.
    """

    def test_calculate_diffs(self):
        """The function 'calculate_diffs' should return a tuple of the form
        (diff, pile, card) in diff assending order"""

        hand = [2, 3, 4, 5]
        piles = {1: 1, 2: 1, 3: 99, 4: 99}
        assert calculate_diffs(piles, hand) == [
            (1, 1, 2),
            (2, 1, 3),
            (3, 1, 4),
            (4, 1, 5),
        ]

        hand = [4, 5, 6, 95]
        piles = {1: 1, 2: 1, 3: 99, 4: 99}
        assert calculate_diffs(piles, hand) == [
            (3, 1, 4),
            (4, 1, 5),
            (4, 3, 95),
            (5, 1, 6),
        ]

        hand = [2, 11, 78, 98]
        piles = {1: 1, 2: 10, 3: 79, 4: 99}
        assert calculate_diffs(piles, hand) == [
            (1, 1, 2),
            (1, 2, 11),
            (1, 3, 78),
            (1, 4, 98),
        ]

    def test_strategy_n_diff(self):
        """Test the function 'strategy_n_diff'"""

        table = Table()
        player = Player(1)
        player.hand = [2, 3, 4, 6, 9, 13]

        piles_to_play, cards_to_play, end_game = strategy_n_diff(
            player, table, acceptable_diff=0
        )
        assert piles_to_play == [1, 1]
        assert cards_to_play == [2, 3]
        assert not end_game

        piles_to_play, cards_to_play, end_game = strategy_n_diff(
            player, table, acceptable_diff=1
        )
        assert piles_to_play == [1, 1, 1]
        assert cards_to_play == [2, 3, 4]
        assert not end_game

        piles_to_play, cards_to_play, end_game = strategy_n_diff(
            player, table, acceptable_diff=2
        )
        assert piles_to_play == [1, 1, 1, 1]
        assert cards_to_play == [2, 3, 4, 6]
        assert not end_game

        piles_to_play, cards_to_play, end_game = strategy_n_diff(
            player, table, acceptable_diff=3
        )
        assert piles_to_play == [1, 1, 1, 1, 1]
        assert cards_to_play == [2, 3, 4, 6, 9]
        assert not end_game

        table.add_cards([2], [35])
        player.hand = [4, 5, 6, 8, 15, 25]
        piles_to_play, cards_to_play, end_game = strategy_n_diff(
            player, table, acceptable_diff=1
        )
        assert piles_to_play == [2, 2, 2, 2]
        assert cards_to_play == [25, 15, 5, 6]
        assert not end_game

        table.add_cards([3, 4], [78, 99])
        player.hand = [4, 5, 6, 7, 66, 88, 97]
        piles_to_play, cards_to_play, end_game = strategy_n_diff(
            player, table, acceptable_diff=1
        )
        assert piles_to_play == [3, 4]
        assert cards_to_play == [88, 97]
        assert not end_game

        table.add_cards([1, 2, 3, 4], [98, 99, 2, 3])
        player.hand = [4, 5, 6, 7, 66]
        piles_to_play, cards_to_play, end_game = strategy_n_diff(
            player, table, acceptable_diff=1
        )
        assert piles_to_play == []
        assert cards_to_play == []
        assert end_game
