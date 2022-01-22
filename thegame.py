import random

from reinforcement_learning.nn import NeuralNetwork

from strategies.strategy_simple import strategy_simple
from strategies.strategy_n_diff import strategy_n_diff
from strategies.user_interface import ask_player
from strategies.strategy_rl import strategy_rl


class Table:
    """Class to define what is on the table during TheGame.

    Attributes:
        piles (dict): Piles on the table.
        deck (list): Deck of cards.

    """

    def __init__(self):

        # Piles on the table
        self.piles = {1: [1], 2: [1], 3: [100], 4: [100]}

        # Deck of cards
        self.deck = [i for i in range(2, 100)]
        random.shuffle(self.deck)

    def deal(self):
        """Deal one card from the deck."""

        return self.deck.pop()

    def cards_to_play(self):
        """Return the number of cards to play."""

        if not self.deck:
            return 1
        else:
            return 2

    def add_cards(self, piles, cards):
        """Add cards to piles.

        Args:
            piles (list): List pile number.
            cards (list): List of cards to add on the corresponding pile.

        """

        if not len(piles) == len(cards):
            raise ValueError("piles and cards must be the same length")

        for pile, card in zip(piles, cards):

            if pile == 1 or pile == 2:
                if card > self.piles[pile][-1] or card == self.piles[pile][-1] - 10:
                    self.piles[pile].append(card)
                else:
                    raise ValueError("Invalid card")

            elif pile == 3 or pile == 4:
                if card < self.piles[pile][-1] or card == self.piles[pile][-1] + 10:
                    self.piles[pile].append(card)
                else:
                    raise ValueError("Invalid card")

            else:
                raise ValueError("Pile must be 1, 2, 3 or 4")

    def __str__(self):
        return f"\n Piles : {self.piles[1][-1]} - {self.piles[2][-1]} - {self.piles[3][-1]} - {self.piles[4][-1]} \n"


class Player:
    """Class to define a player.

    Attributes:
        hand (list): List of cards in hand.
        n_players (int): Number of player.

    """

    def __init__(self, n):

        self.n_players = n
        self.hand = []

    def draw(self, card):
        """Draw a card and sort card in hand.

        Args:
            card (int): Card to add in hand.

        """

        self.hand.append(card)
        self.hand.sort()

    def play(self, card):
        """Play a card.

        Args:
            card (int): Card played (removed from hand).

        """

        if card in self.hand:
            self.hand.remove(card)
            return True
        else:
            return False

    def __str__(self):
        return f"In hand you have: {self.hand}"


def calculate_points(table, players):
    """Calculate the points."""

    points = 0
    for player in players:
        points += len(player.hand)
    points += len(table.deck)

    return points


def output(text, screen=True, logfile=False):
    """Print text on the screen and/or in a file.

    Args:
        text (str): Text to print.
        screen (bool): Print on the screen.
        logfile (bool): Print in a file.

    """

    if screen:
        print(str(text))

    if logfile:
        pass
        # with open('thegame.log', 'a') as f:
        #     f.write(str(text) + '\n')


def set_TheGame(n_player):
    """Set up the game."""

    table = Table()
    players = [Player(i + 1) for i in range(n_player)]

    if n_player > MAX_PLAYERS:
        raise ValueError("Too many players")

    for player in players:
        for i in range(N_CARDS[n_player]):
            player.draw(table.deal())

    return players, table


def play_TheGame(players, table, strategy, acceptable_diff=1, disp=False):
    """Play the game.

    Args:
        players (list): List of players.
        table (Table): Table of the game.

    """

    if strategy == "strategy_rl":
        nn = NeuralNetwork(len(players))

    while True:

        for player in players:

            output("\n-----------------------------", disp)
            output(f"Player {player.n_players} turn:", disp)
            output(f"You should play minimum {table.cards_to_play()} cards", disp)
            output(table, disp)
            output(player, disp)

            if not player.hand:

                # Check if nobody has cards
                no_cards = True
                for player in players:
                    if player.hand:
                        no_cards = False
                        break
                if no_cards:
                    score = calculate_points(table, players)
                    output(f"\nYour score is {score}", disp)
                    return score

                output("You have no card left", disp)
                continue

            valid_move = False
            while not valid_move:

                if strategy == "user_interface":
                    try:
                        piles, cards, end_game = ask_player(player)
                    except ValueError:
                        valid_move = False
                        continue

                elif strategy == "strategy_simple":
                    piles, cards, end_game = strategy_simple(player, table)

                elif strategy == "strategy_n_diff":
                    piles, cards, end_game = strategy_n_diff(player, table, acceptable_diff)

                elif strategy == "strategy_rl":
                    piles, cards, end_game = strategy_rl(player, table, nn)

                else:
                    raise ValueError("Strategy not found")

                if end_game:
                    score = calculate_points(table, players)
                    output(f"\nYour score is {score}", disp)
                    return score

                if len(cards) < table.cards_to_play():
                    output(f"You should play minimum {table.cards_to_play()} cards", disp)
                    valid_move = False
                    continue

                if len(cards) > len(player.hand):
                    output("You do not have enough cards", disp)
                    valid_move = False
                    continue

                for card in cards:
                    valid = player.play(card)

                table.add_cards(piles, cards)

                if not valid:
                    output(player.hand, disp)
                    output("You do not have that card", disp)
                    valid_move = False
                    continue

                valid_move = True

            while table.deck and len(player.hand) < N_CARDS[len(players)]:
                player.draw(table.deal())


# Rules
MAX_PLAYERS = 5
N_CARDS = {1: 8, 2: 7, 3: 6, 4: 6, 5: 6}


if __name__ == "__main__":

    # Options
    display_output = True
    n_player = 1
    # strategy = 'strategy_simple'
    # strategy = 'user_interface'
    # strategy = "strategy_n_diff"
    strategy = 'strategy_rl'
    acceptable_diff = 10

    # Set TheGame
    players, table = set_TheGame(n_player)

    # Play TheGame
    play_TheGame(players, table, strategy, acceptable_diff, display_output)
