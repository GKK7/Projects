# import the required module
import socket


# check if a website is running properly
def is_running(site):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((site, 80))
        return True
    except:
        return False


# run the main function
if __name__ == "__main__":
    while True:
        site = input('Website to check: ')
        if is_running(f'{site}'):
            print(f"{site} is running!")
        else:
            print(f'There is a problem with {site}!')
        # ask whether or not to repeat the search
        if input("Would You like to check another website(Y/N)? ") in {'n', 'N'}:
            break
