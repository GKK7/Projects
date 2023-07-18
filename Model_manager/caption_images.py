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

            # Save the caption as a text file with the same name as the original image
            image_filename = os.path.basename(image_path)
            caption_filename = os.path.splitext(image_filename)[0] + ".txt"
            caption_path = os.path.join(args.image_dir, caption_filename)
            with open(caption_path, "w") as f:
                f.write(caption)
        except Exception as e:
            print(f'Error generating caption for image {image_path}: {e}')
            continue

    return captions


image_files = os.listdir(args.image_dir)
image_paths = [os.path.join(args.image_dir, filename) for filename in image_files
               if filename.lower().endswith((".jpg", ".jpeg", ".png"))]


generated_captions = predict_step(image_paths)

for caption, image_path in zip(generated_captions, image_paths):
    print(f"Image: {image_path}")
    print(f"Caption: {caption}")
    print()

data_list = []

for filename in os.listdir(args.image_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(args.image_dir, filename)
        caption_filename = filename.split(".")[0] + ".txt"
        caption_path = os.path.join(args.image_dir, caption_filename)  # Get directory of the image path

        try:
            with open(caption_path, "r") as caption_file:
                caption_text = caption_file.read()
        except Exception as e:
            print(f'Error reading caption for image {image_path}: {e}')
            continue

        image_obj = {
            "image_path": image_path,
            "caption": caption_text
        }
        data_list.append(image_obj)

json_data = json.dumps(data_list, indent=4)  # Convert list to JSON string with indentation
file_path = os.path.join(args.image_dir, "image_data.json") # Specify the file path where you want to save the JSON file

with open(file_path, "w") as json_file:
    json_file.write(json_data)

print("Image data and captions converted and saved to JSON file successfully.")
