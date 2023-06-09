import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from pytorch_lightning import seed_everything
from pathlib import Path
from PIL import Image


seed_everything(123)


# Model directory
models = {
    "RV-Gogh": {
        "path": "/home/gkirilov/Checkpoint/Merged",
        "active": True,
        "keyword": "RV-Gogh"
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
            "8k resolution beautiful cozy inviting Hobbit-House Lighthouse steampunk, digital illustration matte painting trending on Artstation",
            "Beautiful fox in the snowy woods on a sunny day, sharp focus, intricate",
            "Blonde haired beautiful girl, in slavic clothing, with fit hourglass body, snowy forest background",
            "A big pioson dart frog under a mushroom, animal, background rain forest",
            "country house garden, outdoor fire pit wood burning, house in the background, highly detailed, extremely detailed, oil on canvas",
            "A painting of Heaven on earth painting",
            "birthday cake cottage in a forest of lollipops, ultra reallistic, fantasy, beautiful, high definition, colourful.",
            "macro of a badger mixed with a honeybee hybrid with wings, pollinating on a colourful flower",
            "Rainy street with two women, digital painting, ultra reallistic, intricate, beautiful, dynamic lighting, ultra detailed, colourful, light reflections.",
            "Strange little creature, dof, bokeh",
            "Spring, flowers, sunrays, sunny evening, mist",
            "meadow, rivers, blue skies, castles, magnificent, light effect, high - definition, unreal engine, beautiful",
            "Table with various cookies, donuts, cakes, coffee cups on pink background",
            "A beautiful teen girl with long hair and a hair bun, a flower in her hair, gentle smile, blue eyes, a character portrait, pre-raphaelitism, enchanting",
            "Cinematic shot of (taxidermy bird:1.2) inside a glass ball, antique shop, glass, crystal, flowers, moss, rocks, luxurious, terrarium",
            "Inside a Swiss chalet, snow capped mountain view out of window, with a fireplace, at night, interior design, d & d concept art, d & d wallpaper, warm, art, extremely detailed, intricate, beautiful, colourful.",
            "A beautiful woman",
            "The face of happiness",
            "A beautiful wood building with a river side street street. Beautiful in look at its own made-in-air artwork",
            "Temple of the wood fairys, featured on ArtStation, concept art, sharp focus, illustration",
            "8k, best quality, masterpiece, realistic, ultra detail, photography, HDR, ROW photo, highres, absurdres, cinematic light, official art, High-definition, depth of field, (close-up:1.4), slender, cute face, smile, beautiful details eyes, 19years old, pretty, Side braid with ash blonde color,",
            "realistic digital painting of a beautiful woman bathing in a fountain, ice hair, platinum hair, fractal ice crystals, highly detailed, intricate, elegant",
            "meadow, rivers, blue skies, castles, magnificent, light effect, high - definition, unreal engine, beautiful, by Marc Simonetti and Caspar David Friedrich",
            "color photograph, close-up, ((a realistic photo of a beautiful girl)), light, ((glowy skin)), looking_at_viewer, (fit body:1.0), ((medium breasts)), high ponytail, detailed illustration, masterpiece, high quality, realistic, very detailed face",
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