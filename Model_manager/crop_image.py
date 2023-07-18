import os
import argparse
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import smartcrop
import cv2
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(description='Image Cropping Script')
parser.add_argument('--image_dir', type=str, required=True, help='Path to original image directory')
args = parser.parse_args()

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

sc = smartcrop.SmartCrop()

# Create a new directory for the cropped images
output_directory = args.image_dir + "_cropped/"
os.makedirs(output_directory, exist_ok=True)

# List of files in directory
files = os.listdir(args.image_dir)

# Wrap files with tqdm for a progress bar
for filename in tqdm(files, desc='Processing images'):
    try:
        if filename.endswith(".jpg" or ".png"):
            img_path = os.path.join(args.image_dir, filename)
            img = Image.open(img_path).convert("RGB")  # ensure image is RGB
            img_tensor = F.to_tensor(img)

            # If the image is already 512x512, just copy it to the output directory
            if img.size == (512, 512):
                output_filename = os.path.splitext(filename)[0] + '_512.jpg'
                output_path = os.path.join(output_directory, output_filename)
                img.save(output_path)
                print(f'Copied {img_path} to {output_path} without modification')
                continue

            with torch.no_grad():
                prediction = model([img_tensor])

            # bounding boxes that represent different objects
            boxes = prediction[0]['boxes']

            # the highest scoring boxes are selected to be represented in the cropped image
            scores = prediction[0]['scores']

            # Check if any objects were detected
            if boxes.numel() > 0:
                box = boxes[scores.argmax()].numpy()
                box = [int(b) for b in box]
            else:
                print(f"No objects detected in {img_path}, skipping...")
                continue

            image = cv2.imread(img_path)

            if image is None:
                print(f'Failed to load image at {img_path}')
                continue

            # Convert BGR to RGB since cv2 reads images in BGR format by default
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert the numpy image to PIL image
            image = Image.fromarray(image)

            # Make sure the box is within the image boundaries
            x = max(0, box[0])
            y = max(0, box[1])
            w = min(image.width - x, box[2] - box[0])
            h = min(image.height - y, box[3] - box[1])

            # Options for the crop
            options = {
                'width': w,
                'height': h,
                'x': x,
                'y': y
            }

            crop = sc.crop(image, 512, 512, options)
            top_crop = crop['top_crop']

            # Create a cropped image using the bounding box returned by SmartCrop
            crop = image.crop(
                (top_crop['x'], top_crop['y'], top_crop['x'] + top_crop['width'], top_crop['y'] + top_crop['height']))

            # Resize the crop to 512x512
            crop_resized = crop.resize((512, 512))

            # Save the cropped image
            base_filename = os.path.splitext(filename)[0]
            output_filename = base_filename + '_512.jpg'
            output_path = os.path.join(output_directory, output_filename)
            crop_resized.save(output_path)

            print(f'Processed {img_path} and saved cropped version to {output_path}')
        else:
            continue
    except Exception as e:
        print(f'Error processing image {filename}: {e}')
        continue
