# Program used to find the largest file in a location. Faster execution than standard Windows search.
# Looks through all folders and subfolders if a whole drive is selected.

# Import the OS module
import os

path = os.path.abspath(input("Enter path to folder or drive \n"))
size = 0
max_size = 0
largest_file = ""

for folder, subfolders, files in os.walk(path):

    # checking the size of each file
    for file in files:
        size = os.stat(os.path.join(folder, file)).st_size

        # updating maximum size
        if size > max_size:
            max_size = size
            largest_file = os.path.join(folder, file)

max_MB = max_size / (1024 * 1024)

print(f"The largest file is: {largest_file}")
print("Size: {:.2f} MB".format(max_MB))
