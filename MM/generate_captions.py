import os
import argparse
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import json
from termcolor import colored
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*transformers*")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_folder", required=True, help="Path to folder containing images")
    return parser.parse_args()

def load_model():
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model.to(device)
    return model, processor

def process_image(image_path, processor, model):
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")
    inputs = processor(i_image, return_tensors="pt").to(device)
    output_ids = model.generate(**inputs)
    preds = processor.decode(output_ids[0], skip_special_tokens=True)
    return preds.strip()

def process_directory(subdir, image_files, processor, model, root_image_folder, generated_prompts_folder):
    image_paths = [os.path.join(subdir, filename) for filename in image_files]
    if not image_paths:
        return False

    json_filename = os.path.basename(subdir) + ".json"
    file_path = os.path.join(generated_prompts_folder, json_filename)

    if os.path.exists(file_path):
        print(colored(f"\nFile {json_filename} already exists. Skipping...", "magenta"))
        return False

    generated_captions = [process_image(image_path, processor, model) for image_path in tqdm(image_paths, desc="Processing images")]

    data_list = [{"file_name": os.path.relpath(image_path, root_image_folder), "text": caption}
                 for caption, image_path in zip(generated_captions, image_paths)]

    with open(file_path, "w") as json_file:
        json_file.write(json.dumps(data_list, indent=4))

    return True

def main():
    args = get_args()
    model, processor = load_model()

    root_image_folder = args.images_folder

    current_folder = os.getcwd()
    storage_folder = os.path.join(current_folder, "../storage/dreambooth_automation")
    os.makedirs(storage_folder, exist_ok=True)
    generated_prompts_folder = os.path.join(storage_folder, "generated_prompts/json_results/")
    os.makedirs(generated_prompts_folder, exist_ok=True)

    processed_any = False
    for subdir, dirs, files in os.walk(root_image_folder):
        image_files = [file for file in files if file.lower().endswith((".jpg", ".jpeg", ".png"))]
        processed = process_directory(subdir, image_files, processor, model, root_image_folder, generated_prompts_folder)
        if processed:
            processed_any = True

    if processed_any:
        print(colored("\nImage data and captions converted and saved to JSON file successfully using BLIP.", "green"))

if __name__ == "__main__":
    main()