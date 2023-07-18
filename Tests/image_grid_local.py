import os
import numpy as np
from PIL import Image


def make_grid(images, ncols):
    """Make a grid from the images."""
    nrows = np.ceil(len(images) / ncols).astype(int)
    if len(images) % ncols != 0:
        for _ in range(ncols - len(images) % ncols):
            images.append(np.full((770, 770, 3), 255))  # Assuming RGB images with white background and added border
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
    img.save(output_path)
    print(f"Saved image grid in {output_path}")


def process_subdirectory(subdirectory):
    """Process a subdirectory to get all image files."""
    images = []
    files_with_paths = []
    for root, dirs, files in os.walk(subdirectory):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):  # Consider all png and jpg files
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                if img.size not in [(1024, 1024)]:  # ignore images of size 512x512 and 1024x1024
                    files_with_paths.append(img_path)
    files_with_paths.sort()

    sorted_images = []  # list to store images after sorting by size
    for file_path in files_with_paths:
        img = Image.open(file_path)
        img = img.resize((768, 768))  # Adjust the image size
        img_array = np.array(img)
        img_array = np.pad(img_array, ((1, 1), (1, 1), (0, 0)), mode='constant',
                           constant_values=255)  # Add 1-pixel white border
        sorted_images.append(img_array)

    for i in range(0, len(sorted_images), 2):  # assume the images are in pairs
        if i + 1 < len(sorted_images):
            if sorted_images[i].shape < sorted_images[i + 1].shape:
                images.append(sorted_images[i])
                images.append(sorted_images[i + 1])
            else:
                images.append(sorted_images[i + 1])
                images.append(sorted_images[i])
        else:
            images.append(sorted_images[i])

    return images


def main(directory):
    for root, subdirs, files in os.walk(directory):
        for subdir in subdirs:
            subdir_path = os.path.join(root, subdir)
            images = process_subdirectory(subdir_path)
            if images:
                ncols = 6
                grid = make_grid(images, ncols)
                grid_name = f"{os.path.basename(subdir_path)}_image_grid.png"
                output_path = os.path.join(subdir_path, grid_name)
                save_grid(grid, output_path)
            else:
                print(f"No images found in {subdir}")


if __name__ == "__main__":
    directory = os.path.expanduser('/home/gkirilov/Checkpoint/lorazz')
    main(directory)
