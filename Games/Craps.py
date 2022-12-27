import random

print('''Welcome to Craps. The game is played by one player who tosses 2 dice. The player wins if they roll a 7 or 11 
on the first roll, loses if they roll a 2, 3, or 12 on the first roll, and must keep rolling if they roll any other 
number. If the player must keep rolling, the game continues until the player either rolls their point again (in which 
case they win) or rolls a 7 (in which case they lose).''')


def craps():
    # Randomizing 2 dice rolls
    dice1 = random.randint(1, 6)
    dice2 = random.randint(1, 6)

    # The sum of the dice is the player's "point"
    point = dice1 + dice2

    if point == 7 or point == 11:
        print(f"You rolled a {point}, you win!")

    elif point == 2 or point == 3 or point == 12:
        print(f"You rolled a {point}, you win!")

    else:
        print("Your point is", point)
        while True:
            dice1 = random.randint(1, 6)
            dice2 = random.randint(1, 6)
            total = dice1 + dice2
            print(f"You rolled a {total}")
            if total == point:
                print("You win!")
                break
            elif total == 7:
                print("You lose!")
                break


# Playing the game again
def play_again():
    play = input("Want to play Craps again? y/n \n")
    while True:
        if play not in ["y", "n"]:
            print("Invalid input")
            play = input("Want to play Craps again? y/n \n")
            continue
        elif "y" in play:
            craps()
            play_again()
            break
        else:
            print("Thanks for playing")
            break


craps()
play_again()
