import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from pytorch_lightning import seed_everything
from pathlib import Path
from PIL import Image

seed_everything(22234)

# Model directory
model_dir = "/home/gkirilov/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/36a01dc742066de2e8c91e7cf0b8f6b53ef53da1"

pipe = StableDiffusionPipeline.from_pretrained(model_dir, safety_checker=None, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(model_dir, safety_checker=None, torch_dtype=torch.float16)
pipe_img2img = pipe_img2img.to("cuda")

print(f"Model is on: {pipe.device}")
print(f"Image-to-Image model is on: {pipe_img2img.device}")

pipe.enable_xformers_memory_efficient_attention()
pipe_img2img.enable_xformers_memory_efficient_attention()

for main_prompt in [
    "A dog",
    "Snow-White and Rose-Red, Fairy Tale, insanely detailed and intricate dress and jewelry, beautiful face, fantasy, William Holman Hunt, Artgerm, Jim Burns, Intricate, Elegant, Digital Illustration, Scenic, Hyper-Realistic, Hyper-Detailed, 16k, smooth, sharp focus, Artstation, crisp quality.",
]:
    negative = "poorly drawn, ugly, tiling, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy,  writing, calligraphy, sign, cut off"

    out_dir = "/home/gkirilov/Checkpoint/stable-diffusion-2-1/"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    prompt = "(Nice) " + main_prompt
    short_prompt = prompt[0:40]

    image = pipe(prompt, guidance_scale=8, negative_prompt=negative).images[0]
    image.save(out_dir + short_prompt  + "_512.png")

    image = image.resize((768, 768), resample=Image.BOX)
    image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.7, negative_prompt=negative).images[0]
    image.save(out_dir + short_prompt  + "_768.png")

    image = image.resize((1024, 1024), resample=Image.BOX)
    image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.7, negative_prompt=negative).images[0]
    image.save(out_dir + short_prompt  + "_1024.png")