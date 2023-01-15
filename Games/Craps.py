import random

print('''Welcome to Craps. The game is played by one player who tosses 2 dice. The player wins if they roll a 7 or 11 
on the first roll, loses if they roll a 2, 3, or 12 on the first roll, and must keep rolling if they roll any other 
number. If the player must keep rolling, the game continues until the player either rolls their point again (in which 
case they win) or rolls a 7 (in which case they lose). Winning bets pay out double.
Your balance is 100''')



def craps():
    global bet
    stack = 100


    while True:
        try:
            bet = int(input("Enter your bet: "))
            if bet > stack:
                print("You don't have enough money to place that bet!")
                continue
            else:
                pass
        except ValueError:
            print("Invalid bet, try again")
            continue
        else:
            stack = stack - bet
            break

    # Randomizing 2 dice rolls
    dice1 = random.randint(1, 6)
    dice2 = random.randint(1, 6)

    # The sum of the dice is the player's "point"
    point = dice1 + dice2
    win_bet = bet * 2 + stack

    # Win and Lose scenarios
    if point == 7 or point == 11:
        print(f"You rolled a {point}, you win! You have {win_bet}")

    elif point == 2 or point == 3 or point == 12:
        print(f"You rolled a {point}, you win! You have {win_bet}")

    else:
        print("Your point is", point)
        while True:
            dice1 = random.randint(1, 6)
            dice2 = random.randint(1, 6)
            total = dice1 + dice2
            print(f"You rolled a {total}")
            if total == point:
                print(f"You win! You have {win_bet}")
                break
            elif total == 7:
                print(f"You lose! You have {stack}")
                break


craps()


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


play_again()
