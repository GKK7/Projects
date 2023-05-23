import subprocess
import torch
import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from pytorch_lightning import seed_everything
from pathlib import Path
from PIL import Image
import train_dreambooth

torch.cuda.set_per_process_memory_fraction(0.8)
seed_everything(123)

models = {
    "RV1.4": {
        "path": "/home/gkirilov/Jenna_Ortega_test_RV1.4",
        "active": True,
        "keyword": "JO 1.4"
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
            "woman, solo, light blue hair, dark blue eyes, detailed face, ([Julianne Hough|Megan Fox|Christina Hendricks]:0.8), (puffy lips :0.9), masterpiece, professional, high quality, beautiful, amazing, gothic, Getty Images, miko, giant, photoshoot, 4k, realistic",
            "woman, solo, JENNAORTEGAX, light blue hair, dark blue eyes, detailed face, ([Julianne Hough|Megan Fox|Christina Hendricks]:0.8), (puffy lips :0.9), masterpiece, professional, high quality, beautiful, amazing, gothic, Getty Images, miko, giant, photoshoot, 4k, realistic",
            "RAW photo, portrait photo of a woman, black hair,  wearing black clothes, dark eyes, JENNAORTEGAX, professional photography detailed, soft lightning, high quality, Fujifilm XT3",
            "photo of 42 y.o man in black clothes, bald, face, half body, body, high detailed skin, skin pores, coastline, overcast weather, wind, waves, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
            "close up photo of a rabbit, forest, haze, halation, bloom, dramatic atmosphere, centred, rule of thirds, 200mm 1.4f macro shot",
            "A stunning intricate full color face portrait of JENNAORTEGAX, wearing a black turtleneck, epic character composition, by ilya kuvshinov, alessio albi, nina masic, sharp focus, natural lighting, subsurface scattering, f4, 35mm, film grain, best shadow",
            "waist up portrait photo, JENNAORTEGAX, in (garden:1.1), posing, wearing sexy spiderman suit, (solo:1.2)"
            ]:
            negative = "poorly drawn, ugly, nude, tiling, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"

            out_dir = f"/home/gkirilov/Checkpoint/{model_info['keyword']}/"
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            prompt = "(" + model_info["keyword"] + ") " + main_prompt
            short_prompt = prompt[0:40]

            image = pipe(prompt, guidance_scale=8, negative_prompt=negative).images[0]
            image.save(out_dir + short_prompt + "_JORTEGA1.4_512.png")

            image = image.resize((768, 768), resample=Image.BOX)
            image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.7, negative_prompt=negative).images[0]
            image.save(out_dir + short_prompt + "_JORTEGA1.4_768.png")

            image = image.resize((1024, 1024), resample=Image.BOX)
            image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.7, negative_prompt=negative).images[0]
            image.save(out_dir + short_prompt + "_JORTEGA1.4_1024.png")

torch.cuda.empty_cache()