import sys
import pytest

sys.path.append("..")

from thegame import Table, Player


class TestTable:
    """
    Test the main function of the class 'Table'.
    """

    table = Table()

    def test_init(self):

        assert self.table.piles[1] == [1]
        assert self.table.piles[2] == [1]
        assert self.table.piles[3] == [100]
        assert self.table.piles[4] == [100]

        assert len(self.table.deck) == 98
        assert not self.table.deck == [i for i in range(2, 100)]

    def test_deal(self):

        assert self.table.deck[-1] == self.table.deal()

        for _ in range(9):
            self.table.deal()

        assert len(self.table.deck) == 88

    def test_cards_to_play(self):

        assert self.table.cards_to_play() == 2

        for _ in range(88):
            self.table.deal()

        assert self.table.cards_to_play() == 1

    def test_add_cards(self):

        # Test not same length between piles and card
        with pytest.raises(ValueError):
            self.table.add_cards([2, 3], [1])

        # Test not existing pile
        with pytest.raises(ValueError):
            self.table.add_cards([1, 1, 5], [2, 3, 4])

        # Valid card
        self.table.add_cards([1], [22])
        self.table.add_cards([1], [12])
        self.table.add_cards([4], [78])
        self.table.add_cards([4], [68])
        assert self.table.piles[1][-1] == 12
        assert self.table.piles[4][-1] == 68

        # Test invalid card
        with pytest.raises(ValueError):
            self.table.add_cards([1], [11])
        with pytest.raises(ValueError):
            self.table.add_cards([4], [70])


class TestPlayer:
    """
    Test the main function of the class 'Player'.
    """

    player = Player(1)

    def test_init(self):

        assert self.player.n_players == 1
        assert self.player.hand == []

    def test_draw(self):

        self.player.draw(5)
        self.player.draw(35)
        self.player.draw(17)

        assert self.player.hand == [5, 17, 35]

    def test_play(self):

        assert self.player.play(5)
        assert self.player.hand == [17, 35]

        assert not self.player.play(5)
        assert not self.player.play(94)
