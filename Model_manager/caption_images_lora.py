import os
import argparse
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Set up argparse
parser = argparse.ArgumentParser(description='Image Captioning Script')
parser.add_argument('--image_dir', type=str, required=True, help='Path to image directory')
args = parser.parse_args()

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image_paths):
    captions = []
    for image_path in tqdm(image_paths, desc='Generating captions'):
        try:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

            output_ids = model.generate(pixel_values, **gen_kwargs)

            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            caption = preds[0].strip()
            captions.append(caption)
        except Exception as e:
            print(f'Error generating caption for image {image_path}: {e}')
            continue
    return captions

image_files = os.listdir(args.image_dir)
image_paths = [os.path.join(args.image_dir, filename) for filename in image_files if filename.lower().endswith((".jpg", ".jpeg", ".png"))]
generated_captions = predict_step(image_paths)

file_path = os.path.join(args.image_dir, "metadata.jsonl")  # specify .jsonl extension

# Open the JSONL file in write mode
with open(file_path, "w") as jsonl_file:
    for caption, image_path in zip(generated_captions, image_paths):
        print(f"Image: {image_path}")
        print(f"Caption: {caption}")
        print()

        # Define file_name as relative path from image_folder
        file_name = os.path.relpath(image_path, args.image_dir)

        image_obj = {
            "file_name": file_name,
            "text": caption
        }

        # Convert dict to JSON then write it to the file with a newline
        try:
            jsonl_file.write(json.dumps(image_obj) + "\n")
        except Exception as e:
            print(f'Error writing to JSONL file for image {image_path}: {e}')
            continue

print("Image data and captions converted and saved to JSONL file successfully.")
