import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from pytorch_lightning import seed_everything
from pathlib import Path
from PIL import Image

seed_everything(123)

# Model directory
models = {
    "Jennalba_400_4e-06": {
        "path": "/home/gkirilov/Brand_new_PC/jenna_mix/dreambooth_dreamshaper_6BakedVae_steps_400_lr_4e-06_batch_1",
        "active": True,
        "keyword": "Jennalba_400_4e-06"
    },
    "Jennalba_800_4e-06": {
        "path": "/home/gkirilov/Brand_new_PC/jenna_mix/dreambooth_dreamshaper_6BakedVae_steps_800_lr_4e-06_batch_1",
        "active": True,
        "keyword": "Jennalba_800_4e-06"
    },
    "Jennalba-1200-4e-06": {
        "path": "/home/gkirilov/Brand_new_PC/jenna_mix/dreambooth_dreamshaper_6BakedVae_steps_1200_lr_4e-06_batch_1",
        "active": True,
        "keyword": "Jennalba-1200-4e-06"
    },
    "Jennalba-1600-4e-06": {
        "path": "/home/gkirilov/Brand_new_PC/jenna_mix/dreambooth_dreamshaper_6BakedVae_steps_1600_lr_4e-06_batch_1",
        "active": True,
        "keyword": "Jennalba-1600-4e-06"
    },
    "Jennalba_2000-4e-06": {
        "path": "/home/gkirilov/Brand_new_PC/jenna_mix/dreambooth_dreamshaper_6BakedVae_steps_2000_lr_4e-06_batch_1",
        "active": True,
        "keyword": "Jennalba_2000-4e-06"
    },
    "Jennalba_2500-4e-06": {
        "path": "/home/gkirilov/Brand_new_PC/jenna_mix/dreambooth_dreamshaper_6BakedVae_steps_2500_lr_4e-06_batch_1",
        "active": True,
        "keyword": "Jennalba_2500-4e-06"
    },
    "Jennalba_3000-4e-06": {
        "path": "/home/gkirilov/Brand_new_PC/jenna_mix/dreambooth_dreamshaper_6BakedVae_steps_3000_lr_4e-06_batch_1",
        "active": True,
        "keyword": "Jennalba_3000-4e-06"
    },
    "Jennalba_4000-4e-06": {
        "path": "/home/gkirilov/Brand_new_PC/jenna_mix/dreambooth_dreamshaper_6BakedVae_steps_4000_lr_4e-06_batch_1",
        "active": True,
        "keyword": "Jennalba_4000-4e-06"
    },
    "Jennalba_5000-4e-06": {
        "path": "/home/gkirilov/Brand_new_PC/jenna_mix/dreambooth_dreamshaper_6BakedVae_steps_5000_lr_4e-06_batch_1",
        "active": True,
        "keyword": "Jennalba_5000-4e-06"
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
            "woman, solo, black hair, dark blue eyes, detailed face, (puffy lips :0.9), masterpiece, professional, high quality, beautiful, amazing, gothic, Getty Images, miko, giant, photoshoot, 4k, realistic",
            "woman, solo, JENNAORTEGAX, black hair, dark blue eyes, detailed face, (puffy lips :0.9), masterpiece, professional, high quality, beautiful, amazing, gothic, Getty Images, miko, giant, photoshoot, 4k, realistic",
            "woman, solo, (JENNAORTEGAX:1), black hair, dark blue eyes, detailed face, (puffy lips :0.9), masterpiece, professional, high quality, beautiful, amazing, gothic, Getty Images, miko, giant, photoshoot, 4k, realistic",
            "A big pioson dart frog under a mushroom, animal, background rain forest, 3d uv colors, ultra reallistic, extremely detailed, 8k, poster, ultra detailed,high definition, colourful",
            "RAW photo, portrait photo of a woman, black hair,  wearing black clothes, dark eyes, JENNAORTEGAX, professional photography detailed, soft lightning, high quality, Fujifilm XT3",
            "RAW photo, portrait photo of a woman, black hair,  wearing black clothes, dark eyes, professional photography detailed, soft lightning, high quality, Fujifilm XT3",
            "photo of 42 y.o man in black clothes, bald, face, half body, body, high detailed skin, skin pores, coastline, overcast weather, wind, waves, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
            "A stunning intricate full color face portrait of JENNAORTEGAX, wearing a black turtleneck, epic character composition, by ilya kuvshinov, alessio albi, nina masic, sharp focus, natural lighting, subsurface scattering, f4, 35mm, film grain, best shadow",
            "A stunning intricate full color face portrait of a beautiful woman, wearing a black turtleneck, epic character composition, by ilya kuvshinov, alessio albi, nina masic, sharp focus, natural lighting, subsurface scattering, f4, 35mm, film grain, best shadow",
            "waist up portrait photo of a woman, JENNAORTEGAX, in (garden:1.1), posing, wearing a spiderman suit, (solo:1.2)",
            "waist up portrait photo of a woman, beautiful face, in (garden:1.1), posing, wearing a spiderman suit, (solo:1.2)",
            "Rainy street with two women by Leonid Afremov, digital painting, ultra reallistic, intricate, beautiful, dynamic lighting, ultra detailed, colourful, light reflections.",
            "8k, best quality, JENNAORTEGAX, masterpiece, realistic, ultra detail, photography, HDR, ROW photo, highres, absurdres, cinematic light, official art, High-definition, depth of field, (close-up:1.4), slender, cute face, smile, beautiful details eyes, 19years old, pretty, Side braid with ash blonde color,",
            "8k, best quality, masterpiece, realistic, ultra detail, photography, HDR, ROW photo, highres, absurdres, cinematic light, official art, High-definition, depth of field, (close-up:1.4), slender, cute face, smile, beautiful details eyes, 19years old, pretty, Side braid with ash blonde color,",
            "realistic digital painting of a beautiful woman bathing in a fountain, ice hair, platinum hair, fractal ice crystals, highly detailed, intricate, elegant",
            "meadow, rivers, blue skies, castles, magnificent, light effect, high - definition, unreal engine, beautiful, by Marc Simonetti and Caspar David Friedrich",
            "color photograph, JENNAORTEGAX, close-up, ((a realistic photo of a beautiful girl)), light, ((glowy skin)), looking_at_viewer, (fit body:1.0), ((medium breasts)), high ponytail, detailed illustration, masterpiece, high quality, realistic, very detailed face",
            "color photograph, close-up, ((a realistic photo of a beautiful girl)), light, ((glowy skin)), looking_at_viewer, (fit body:1.0), ((medium breasts)), high ponytail, detailed illustration, masterpiece, high quality, realistic, very detailed face",
            "A beautiful teen girl with long hair and a hair bun, a flower in her hair, gentle smile, blue eyes, a character portrait, pre-raphaelitism, studio photograph, enchanting",
            "Inside a Swiss chalet, snow capped mountain view out of window, with a fireplace, at night, interior design, d & d concept art, d & d wallpaper, warm, digital art. art by james gurney and larry elmore, extremely detailed, intricate, beautiful, colourful.",
            "Seductive lingerie, portrait, luminous necklace, luminous butterflies, film grain, closup, focus blur"
        ]:
            negative = "(worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), poorly drawn, low resolution, ugly, nude, tiling, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"

            out_dir = f"/home/gkirilov/Checkpoint/{model_info['keyword']}/"
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            prompt = "(" + model_info["keyword"] + ") " + main_prompt
            short_prompt = prompt[0:40]

            image = pipe(prompt, guidance_scale=8, negative_prompt=negative).images[0]
            image.save(out_dir + short_prompt  + "_512.png")

            image = image.resize((768, 768), resample=Image.BOX)
            image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.65, negative_prompt=negative).images[0]
            image.save(out_dir + short_prompt  + "_768.png")

            image = image.resize((1024, 1024), resample=Image.BOX)
            image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.55, negative_prompt=negative).images[0]
            image.save(out_dir + short_prompt  + "_1024.png")



torch.cuda.empty_cache()