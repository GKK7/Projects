from random import randint


# Creating a Roulette class that randomizes rolls, takes bets and error proofs
class Roulette:
    def __init__(self):
        self.numlist = []
        print("Welcome to Roulette, pick numbers and place bets")
        # Ask for how many numbers (up to 36) the player would like to bet on
        while True:
            try:
                self.numbers = int(input("How many numbers would you like to bet on?\n"))
                if self.numbers > 36:
                    print("Less than 36 please")
                    continue
            except ValueError:
                print("Integers numbers only")
            else:
                break
        for x in range(0, self.numbers):
            # Ask for all numbers (up to 36) the player would like to bet on 
            while True:
                try:
                    self.s2 = int(input("Pick a number from 0 to 36: "))
                    if self.s2 > 36:
                        print("Must be 36 or lower")
                        continue
                except ValueError:
                    print("Integers only")
                    continue
                else:
                    self.numlist.append(self.s2)
                    break
        print(f"Your numbers are {self.numlist}")

    # Creating a chips function
    def chips(self):
        self.stack = 100
        while True:
            try:
                self.bet = float(input(f"You have {self.stack}, what's your bet?"))
                if self.bet > self.stack:
                    print("Not enough chips")
                    continue
                else:
                    pass
            except ValueError:
                print("Invalid bet, try again")
                continue
            else:
                self.stack = self.stack - self.bet
                break

    # Spinning the wheel
    def spin_roulette(self):
        self.spin = int(randint(0, 36))
        print(f"The roulette spins and lands on: {self.spin}")

    # Calculating outcome
    def outcome(self):
        if self.spin not in self.numlist:
            print("You lose")
            print(f"Your balance is: {self.stack}")
        else:
            print("You win")
            self.stack = self.stack + (1 / self.numbers) * 36 * self.bet
            print(f"You have {self.stack} chips")


Result = Roulette()
Result.chips()
Result.spin_roulette()
Result.outcome()


# Playing the game again
def play_again():
    new_game = input("Would you like to play again? y/n").lower()
    while True:
        if new_game not in ["y", "n"]:
            print("Invalid input")
            new_game = input("Would you like to play again? y/n").lower()
            continue
        elif "y" in new_game:
            test = Roulette()
            test.chips()
            test.spin_roulette()
            test.outcome()
            play_again()
            break
        else:
            print("Thanks for playing")
            break


play_again()
