# Describing game options
class Gamemaster:
    def __init__(self):
        print(""
              "Welcome. The game options are:\n"
              "1. Roulette\n"
              "2. Hangman\n"
              "3. Slots\n"
              "4. Craps\n"
              "5. Exit\n")

    choices = {1: "Roulette", 2: "Hangman", 3: "Slots", 4: "Craps", 5: "Exit"}

    # Choosing a game and error proofing
    def game_choice(self):
        while True:
            try:
                choice = int(input("Select the number of the game you would like to play or 5 to quit\n"))
                if choice not in self.choices.keys():
                    print("Unavailable option")
                    continue
            except ValueError:
                print("Enter a number from 1 to 5 please. Integers only")
                continue
            else:
                print(f"You have chosen {self.choices[choice]}")
                if choice == 1:
                    import Roulette
                if choice == 2:
                    import Hangman
                if choice == 3:
                    import Slots
                if choice == 4:
                    import Craps
                if choice == 5:
                    quit()
                    break


game_select = Gamemaster()
game_select.game_choice()
