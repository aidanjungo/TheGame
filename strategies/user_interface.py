def ask_player(player):

    valid_cards = False
    valid_piles = False

    while not valid_cards:

        print('\nType "Quit", "Q" or "q" if you cannot play!')

        cards = input("Enter list of cards (coma serparaded): ")

        if cards in ["Quit", "Q", "q"]:
            return False, False, True

        cards = [int(card) for card in cards.split(",")]

        for card in cards:
            if card not in player.hand:
                print("You do not have that card")
                valid_cards = False
                raise ValueError("You do not have that card")
            valid_cards = True

    while not valid_piles:

        piles = input("Enter pile number (coma separated): ")
        piles = [int(pile) for pile in piles.split(",")]

        if len(piles) != len(cards):
            continue
        for pile in piles:
            if pile not in [1, 2, 3, 4]:
                valid_piles = False
                raise ValueError("Pile is not correct! Pile must be 1, 2, 3 or 4")
            valid_piles = True

    return piles, cards, False
