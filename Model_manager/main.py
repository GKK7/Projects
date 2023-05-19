import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from pytorch_lightning import seed_everything
from pathlib import Path
from PIL import Image
import json
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

# Captioning from a cached HF model named nlpconnect/vit-gpt2-image-captioning
directory = "/home/gkirilov/Jenna_Ortega_resized"
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Run on GPU with specified parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Caption the images in the designated folder
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

        # Save the caption as a text file with the same name as the original image
        image_filename = os.path.basename(image_path)
        caption_filename = os.path.splitext(image_filename)[0] + ".txt"
        caption_path = os.path.join("/home/gkirilov/Jenna_Ortega_resized", caption_filename)
        with open(caption_path, "w") as f:
            f.write(caption)

    return captions


image_folder = "/home/gkirilov/Jenna_Ortega_resized"
image_files = os.listdir(image_folder)
image_paths = [os.path.join(image_folder, filename) for filename in image_files if filename.endswith(".jpg")]
generated_captions = predict_step(image_paths)

data_list = []
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        image_path = os.path.join(directory, filename)
        caption_path = os.path.join(directory, filename.split(".")[0] + ".txt")

        with open(caption_path, "r") as caption_file:
            caption_text = caption_file.read()

        image_obj = {
            "image_path": image_path,
            "caption": caption_text
        }
        data_list.append(image_obj)


json_data = json.dumps(data_list, indent=4)
file_path = "/home/gkirilov/Jenna_Ortega_resized/image_data.json"

with open(file_path, "w") as json_file:
    json_file.write(json_data)

print("Image data and captions converted and saved to JSON file successfully.")

seed_everything(123)

# List cached text-2-image models that are in the local model directory
models = {
    "Stable Diffusion 2.1": {
        "path": "/home/gkirilov/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/36a01dc742066de2e8c91e7cf0b8f6b53ef53da1",
        "active": False,
        "keyword": "SD 2.1"
    },
    "Dreamlike Photoreal 2.0": {
        "path": "/home/gkirilov/.cache/huggingface/hub/models--dreamlike-art--dreamlike-photoreal-2.0/snapshots/d9e27ac81cfa72def39d74ca673219c349f0a0d5",
        "active": False,
        "keyword": "Dreamlike Photoreal 2.0"
    },
    "Prompthero: Open journey": {
        "path": "/home/gkirilov/.cache/huggingface/hub/models--prompthero--openjourney/snapshots/e291118e93d5423dc88ac1ed93c02362b17d698f",
        "active": False,
        "keyword": "Prompthero: Open journey"
    },
    "Stable Diffusion 1.5": {
        "path": "/home/gkirilov/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/aa9ba505e1973ae5cd05f5aedd345178f52f8e6a",
        "active": True,
        "keyword": "SD 1.5"
    },
    "Redshift Diffusion 768": {
        "path": "/home/gkirilov/.cache/huggingface/hub/models--nitrosocke--redshift-diffusion-768/snapshots/30b5fd6173c924f9d63a205efd505843938d672d",
        "active": False,
        "keyword": "Redshift 768"
    },
    "Realistic Vision 1.4": {
        "path": "/home/gkirilov/.cache/huggingface/hub/models--SG161222--Realistic_Vision_V1.4/snapshots/6e02ccb36bd0d45ec803cd58cd61f7e7edaff39f",
        "active": False,
        "keyword": "Realvision 1.4"
    },
}

for model_name, model_info in models.items():
    if model_info["active"]:
        pipe = StableDiffusionPipeline.from_pretrained(model_info["path"], safety_checker=None, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(model_info["path"], safety_checker=None, torch_dtype=torch.float16)
        pipe_img2img = pipe_img2img.to("cuda")

        print(f"Model {model_name} is on: {pipe.device}")
        print(f"Image-to-Image model {model_name} is on: {pipe_img2img.device}")

        pipe.enable_xformers_memory_efficient_attention()
        pipe_img2img.enable_xformers_memory_efficient_attention()

torch.cuda.empty_cache()