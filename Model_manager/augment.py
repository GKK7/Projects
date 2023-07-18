import os
import argparse
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(description='Image Augmentation Script')
parser.add_argument('--image_dir', type=str, required=True, help='Path to original image directory')
args = parser.parse_args()

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomVerticalFlip(p=0.0),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
])

# Create a new folder with the original name + "-augmented"
augmented_data_dir = os.path.join(args.image_dir, "augmented")
os.makedirs(augmented_data_dir, exist_ok=True)

# Get the list of files
file_list = os.listdir(args.image_dir)

# Go through all the images in the original data directory
for filename in tqdm(file_list, desc="Processing images", unit="image"):
    try:
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(args.image_dir, filename))
            for i in range(3):
                transformed_img = transform(img)
                new_filename = f"{os.path.splitext(filename)[0]}_{i}{os.path.splitext(filename)[1]}"
                transformed_img.save(os.path.join(augmented_data_dir, new_filename))
    except Exception as e:
        print(f"Skipping {filename} due to an error: {str(e)}")

print("Images have been augmented and saved to the augmented directory.")
