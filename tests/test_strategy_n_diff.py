import sys

sys.path.append("..")

from strategies.strategy_n_diff import strategy_n_diff, calculate_diffs


class TestStrategy:
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

        hand = [2, 3, 4, 5]
        piles = {1: 1, 2: 1, 3: 99, 4: 99}
        assert calculate_diffs(piles, hand) == [
            (1, 1, 2),
            (2, 1, 3),
            (3, 1, 4),
            (4, 1, 5),
        ]
