import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from pytorch_lightning import seed_everything
from pathlib import Path
from PIL import Image

seed_everything(123)


# Model directory
models = {
    "Alvan 1.0": {
        "path": "/home/gkirilov/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/aa9ba505e1973ae5cd05f5aedd345178f52f8e6a",
        "active": True,
        "keyword": "Avan1.0"
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

        for main_prompt in [
            "A photo of a dog in the park, high-detail",
        ]:
            negative = "poorly drawn, ugly, tiling, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"

            out_dir = f"/home/gkirilov/Checkpoint/{model_info['keyword']}/"
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            prompt = "(" + model_info["keyword"] + ") " + main_prompt
            short_prompt = prompt[0:40]

            image = pipe(prompt, guidance_scale=5, negative_prompt=negative, num_images_per_prompt=10).images[0]
            image.save(out_dir + short_prompt  + "_512.png")

            image = image.resize((768, 768), resample=Image.BOX)
            image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.7, negative_prompt=negative).images[0]
            image.save(out_dir + short_prompt  + "_768.png")

            image = image.resize((1024, 1024), resample=Image.BOX)
            image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.7, negative_prompt=negative).images[0]
            image.save(out_dir + short_prompt  + "_1024.png")



torch.cuda.empty_cache()