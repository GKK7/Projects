# Describing game options
class Gamemaster:
    def __init__(self):
        print(""
              "Welcome. The game options are:\n"
              "1. Roulette\n"
              "2. Blackjack\n"
              "3. Hangman\n"
              "4. Slots\n"
              "5. Craps\n"
              "6. Exit\n")

    choices = {1: "Roulette", 2: "Blackjack", 3: "Hangman", 4: "Slots", 5: "Craps", 6: "Exit"}

    # Choosing a game and error proofing
    def game_choice(self):
        while True:
            try:
                choice = int(input("Select the number of the game you would like to play or 6 to quit\n"))
                if choice not in self.choices.keys():
                    print("Unavailable option")
                    continue
            except ValueError:
                print("Enter a number from 1 to 6 please. Integers only")
                continue
            else:
                print(f"You have chosen {self.choices[choice]}")
                if choice == 1:
                    import Roulette
                if choice == 2:
                    import BlackJack
                if choice == 3:
                    import Hangman
                if choice == 4:
                    import Slots
                if choice == 5:
                    import Craps
                if choice == 6:
                    quit()
                    break


game_select = Gamemaster()
game_select.game_choice()
