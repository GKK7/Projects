# import the required module
import pyshorteners


# run the shorten link program
def shorten():
    link = input("Link to shorten: \n")
    short = pyshorteners.Shortener()
    new_link = short.tinyurl.short(link)
    print(f"Shortened link is:\n{new_link}")


shorten()
