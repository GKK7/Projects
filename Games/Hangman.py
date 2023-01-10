# Importing the required english-words module that will pick random words

from english_words import english_words_set
import random

# Ignoring DeprecationWarning that might occur depending on python version

import warnings

warnings.simplefilter(action='ignore', category=DeprecationWarning)

# Randomizing the word

randword = random.sample([i for i in english_words_set if len(i) < 8], 1)
final_word = "".join(randword).lower().strip()


# Guessing the word

def guess_word():
    print(f"Welcome to Hangman. Try to guess the 8 or lower letter word. \nThis one has {len(final_word)} letters")
    letters_guessed = ""
    lives = 6
    while lives > 0:
        guess = input(" Guess a letter:\n").lower()
        if guess in final_word and len(guess) < 2:
            print("Correct")
        elif len(guess) >= 2:
            print("One letter max, try again")
            continue
        elif not guess.isalpha():
            print("Letters only please")
        else:
            lives -= 1
            print("Incorrect")
        letters_guessed = letters_guessed + guess
        print(f"Lives count is {lives}")

        wrongletter = 0
        for letter in final_word:
            if letter in letters_guessed:
                print(f"{letter}", end="")
            else:
                print("_", end="")
                wrongletter += 1

        if wrongletter == 0:
            print(f"\nYou win, the word is {final_word}")
            break
    else:
        print(f"\nYou lost, the word is {final_word}")


guess_word()


# Playing the game again

def play_again():
    play = input("Want to play Hangman again? y/n \n")
    while True:
        if play not in ["y", "n"]:
            print("Invalid input")
            play = input("Want to play Hangman again ? y/n \n")
            continue
        elif "y" in play:
            guess_word()
            play_again()
            break
        else:
            print("Thanks for playing")
            break


play_again()
