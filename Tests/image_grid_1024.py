import os
import numpy as np
from PIL import Image
import argparse
import shutil
from termcolor import colored


def make_grid(images, ncols):
    """Make a grid from the images."""
    nrows = np.ceil(len(images) / ncols).astype(int)
    if len(images) % ncols != 0:
        for _ in range(ncols - len(images) % ncols):
            images.append(np.full((1026, 1026, 3), 255))  # Assuming RGB images with white background and added border
    grid = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            row.append(images[i * ncols + j])
        grid.append(np.concatenate(row, axis=1))
    grid = np.concatenate(grid, axis=0)
    return grid


def save_grid(grid, output_path):
    """Save the grid as an image."""
    img = Image.fromarray(grid.astype('uint8'))
    # Replace .png extension with .jpg
    output_path = output_path.replace('.png', '.jpg')
    img.save(output_path, 'JPEG', quality=90)
    print(colored(f"Saved image grid in {output_path}\n", "green"))


def process_subdirectory(subdirectory):
    """Process a subdirectory to get all image files."""
    images = []
    files_with_paths = []
    for root, dirs, files in os.walk(subdirectory):
        for file in files:
            if file.endswith(".png") and '512' not in file and '768' not in file:  # ignore files with '512' in their name
                img_path = os.path.join(root, file)
                files_with_paths.append(img_path)
    files_with_paths.sort()

    sorted_images = []  # list to store images after sorting by size
    for file_path in files_with_paths:
        img = Image.open(file_path)
        img = img.resize((1024, 1024))  # Adjust the image size
        img_array = np.array(img)
        img_array = np.pad(img_array, ((1, 1), (1, 1), (0, 0)), mode='constant',
                           constant_values=255)  # Add 1-pixel white border
        sorted_images.append(img_array)

    for i in range(0, len(sorted_images), 2):  # assume the images are in pairs
        if i + 1 < len(sorted_images):  # make sure the next index exists
            if sorted_images[i].shape < sorted_images[i + 1].shape:  # if 768x768 image comes before 1024x1024
                images.append(sorted_images[i])
                images.append(sorted_images[i + 1])
            else:  # if 1024x1024 image comes before 768x768
                images.append(sorted_images[i + 1])
                images.append(sorted_images[i])
        else:
            images.append(sorted_images[i])  # append the last image if number of images is not even

    return images


def main(directory):
    """Main function."""
    for root, subdirs, files in os.walk(directory):
        for subdir in subdirs:
            subdir_path = os.path.join(root, subdir)
            images = process_subdirectory(subdir_path)
            if images:
                ncols = 4  # change the number of columns back to 6
                grid = make_grid(images, ncols)
                grid_name = f"{os.path.basename(subdir_path)}_image_grid.png"  # change file name here
                output_path = os.path.join(subdir_path, grid_name)
                save_grid(grid, output_path)
            else:
                print(f"No images found in {subdir}")


if __name__ == "__main__":
    directory = '/home/gkirilov/auto_dreambooth_v3/generated_images'  # Put your directory path here
    main(directory)