import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
from pytorch_lightning import seed_everything
from diffusers import StableDiffusionDepth2ImgPipeline

import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
import random

import xformers
from accelerate import Accelerator

seed_everything(22234)


for model_id, keyword in {
    # "nitrosocke/redshift-diffusion-768": "redshift style",
    # "stabilityai/stable-diffusion-2-1": "Nice",
    # "prompthero/openjourney": "mdjrny-v4 style",
    # "nitrosocke/redshift-diffusion": "redshift style",
    # "nitrosocke/Future-Diffusion": "future style",
    # "nitrosocke/mo-di-diffusion": "modern disney style",
    "dreamlike-art/dreamlike-photoreal-2.0": "dreamlike-photoreal-2.0",
    "runwayml/stable-diffusion-v1-5": "stable-diffusion-v1-5",
    "SG161222/Realistic_Vision_V1.4": "Realistic_Vision_V1.4",
}.items():

    # pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch.float16, revision="fp16", use_auth_token="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")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, use_auth_token="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",torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, safety_checker=None, use_auth_token="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",torch_dtype=torch.float16)
    pipe_img2img = pipe_img2img.to("cuda")

    print(f"Model is on: {pipe.device}")
    print(f"Image-to-Image model is on: {pipe_img2img.device}")

    # pipe_img2img = StableDiffusionDepth2ImgPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-2-depth",
    #     torch_dtype=torch.float16,
    # ).to("cuda")

    pipe.enable_xformers_memory_efficient_attention()
    pipe_img2img.enable_xformers_memory_efficient_attention()
    # pipe_depth.enable_xformers_memory_efficient_attention()


    for i, main_prompt in enumerate([
        "A dog",
        "Snow-White and Rose-Red, Fairy Tale, insanely detailed and intricate dress and jewelry, beautiful face, fantasy, William Holman Hunt, Artgerm, Jim Burns, Intricate, Elegant, Digital Illustration, Scenic, Hyper-Realistic, Hyper-Detailed, 16k, smooth, sharp focus, Artstation, crisp quality.",

    ]):
        negative="poorly drawn, ugly, tiling, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy,  writing, calligraphy, sign, cut off"
        # negative="mutated body double head bad anatomy long face long neck long body text watermark signature"
        # negative=""

        out_dir = "/home/gkirilov/Checkpoint" + keyword + "/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)


        prompt = "(" + keyword + ") " + main_prompt
        short_prompt = prompt[0:40]

        # image = Image.open('/root/workspace/share/temp/depth/3.png')
        # image = pipe_img2img(prompt=prompt, image=image, strength=0.7,
        #     negative_prompt=negative).images[0]
        # image.save(out_dir + short_prompt  + ".png")



        image = pipe(prompt, guidance_scale=8, negative_prompt=negative).images[0]
        image.save(out_dir + short_prompt  + "_512.png")


        image = image.resize((768, 768), resample=Image.BOX)
        image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.7,
            negative_prompt=negative).images[0]
        image.save(out_dir + short_prompt  + "_768.png")

        image = image.resize((1024, 1024), resample=Image.BOX)
        image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.7,
            negative_prompt=negative).images[0]
        image.save(out_dir + short_prompt  + "_1024.png")

        image = image.resize((1280, 1280), resample=Image.BOX)
        image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.7,
            negative_prompt=negative).images[0]
        image.save(out_dir + short_prompt  + "_1280.png")


        image = image.resize((1536, 1536), resample=Image.BOX)
        image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.7,
            negative_prompt=negative).images[0]
        image.save(out_dir + short_prompt  + "_1536.png")


        # image = image.resize((1792, 1792), resample=Image.BOX)
        # image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.45,
        #     negative_prompt=negative).images[0]
        # image.save(out_dir + short_prompt  + "_1792.png")


        # image = image.resize((2048, 2048), resample=Image.BOX)
        # image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.4,
        #     negative_prompt=negative).images[0]
        # image.save(out_dir + short_prompt  + "_2048.png")

        # image = image.resize((2304, 2304), resample=Image.BOX)
        # image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.35,
        #     negative_prompt=negative).images[0]
        # image.save(out_dir + short_prompt  + "_2304.png")




