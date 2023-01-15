import random

num_slots = 3
num_symbols = 5

symbols = ["#", "^", "%", "&", "*"]


# Generate a random spin
def spin():
    return [random.choice(symbols) for x in range(num_slots)]


# Check if the spin is a winning one
def check_win(spin):
    for symbol in spin:
        if symbol != spin[0]:
            return False
    return True


# Play the game
def play_game():
    print("Welcome to the slots game! You need to hit three of the same symbols to win. Good luck")
    chips = 100
    while True:
        print(f"Your balance is: {chips}")
        try:
            bet = int(input("Enter your bet: "))
            if bet > chips:
                print("You don't have enough money to place that bet!")
                continue
            spin_result = spin()
            print(f"The spin result is: {spin_result}")
            if check_win(spin_result):
                chips += bet
                print(f"You won! You have {chips}")
                break
            else:
                chips -= bet
                print(f"You lost! You have {chips}")
                break
        except ValueError:
            print("Invalid input")


play_game()


# Playing the game again
def play_again():
    play = input("Want to play Slots again? y/n \n")
    while True:
        if play not in ["y", "n"]:
            print("Invalid input")
            play = input("Want to play Slots again? y/n \n")
            continue
        elif "y" in play:
            play_game()
            play_again()
            break
        else:
            print("Thanks for playing")
            break


play_again()
