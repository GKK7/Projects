import os
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import json

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
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(pixel_values, **gen_kwargs)

        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        caption = preds[0].strip()
        captions.append(caption)

    return captions

image_folder = os.path.expanduser('/home/gkirilov/Checkpoint/Jenna_Ortega/dataset/images_50_lora')
image_files = os.listdir(image_folder)
image_paths = [os.path.join(image_folder, filename) for filename in image_files
               if filename.lower().endswith((".jpg", ".jpeg", ".png"))]

generated_captions = predict_step(image_paths)

file_path = os.path.join(image_folder, "metadata.jsonl")  # specify .jsonl extension

# Open the JSONL file in write mode
with open(file_path, "w") as jsonl_file:
    for caption, image_path in zip(generated_captions, image_paths):
        print(f"Image: {image_path}")
        print(f"Caption: {caption}")
        print()

        # Define file_name as relative path from image_folder
        file_name = os.path.relpath(image_path, image_folder)

        image_obj = {
            "file_name": file_name,
            "text": caption
        }

        # Convert dict to JSON then write it to the file with a newline
        jsonl_file.write(json.dumps(image_obj) + "\n")

print("Image data and captions converted and saved to JSONL file successfully.")
