from diffusers import DiffusionPipeline, UNet2DConditionModel
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from transformers import CLIPTextModel
import torch
from pathlib import Path
from PIL import Image
import os

model_id = "/home/gkirilov/Brand_new_PC/van_gogh/dreambooth_dreamshaper_6BakedVae_steps_20000_lr_4e-06_batch_1"

unet_paths = [
    model_id + "/checkpoint-1000/unet",
    model_id + "/checkpoint-2000/unet",
    model_id + "/checkpoint-3000/unet",
    model_id + "/checkpoint-4000/unet",
    model_id + "/checkpoint-5000/unet",
    model_id + "/checkpoint-6000/unet",
    model_id + "/checkpoint-7000/unet",
    model_id + "/checkpoint-8000/unet",
    model_id + "/checkpoint-9000/unet",
    model_id + "/checkpoint-10000/unet",
    model_id + "/checkpoint-11000/unet",
    model_id + "/checkpoint-12000/unet",
    model_id + "/checkpoint-13000/unet",
    model_id + "/checkpoint-14000/unet",
    model_id + "/checkpoint-15000/unet",
    model_id + "/checkpoint-16000/unet",
    model_id + "/checkpoint-17000/unet",
    model_id + "/checkpoint-18000/unet",
    model_id + "/checkpoint-19000/unet",
    model_id + "/checkpoint-20000/unet",
]

base_out_dir = "/home/gkirilov/Checkpoint/van_gogh/"
prompts= [
    "8k resolution beautiful cozy inviting Hobbit-House Lighthouse steampunk, painting by Van Gogh, digital illustration matte painting trending on Artstation",
    "Beautiful fox in the snowy woods on a sunny day painting by Van Gogh, sharp focus, intricate",
    "Blonde haired beautiful girl, in slavic clothing, with fit hourglass body, snowy forest background, painting by Van Gogh",
    "A big pioson dart frog under a mushroom painting by Van Gogh, animal, background rain forest",
    "country house garden painting by Van Gogh, outdoor fire pit wood burning, house in the background, highly detailed, extremely detailed, oil on canvas",
    "A painting of Heaven on earth painting by Van Gogh",
    "birthday cake cottage in a forest of lollipops, ultra reallistic, fantasy, beautiful, high definition, colourful.",
    "macro painting by Van Gogh of a badger mixed with a honeybee hybrid with wings, pollinating on a colourful flower",
    "Rainy street with two women painting by Van Gogh, digital painting, ultra reallistic, intricate, beautiful, dynamic lighting, ultra detailed, colourful, light reflections.",
    "Strange little creature, dof, bokeh, painting by Van Gogh",
    "Spring, flowers, sunrays, sunny evening, mist by Van Gogh",
    "meadow, rivers, blue skies, castles, magnificent, light effect, high - definition, unreal engine, beautiful, painting by Van Gogh, by Van Gogh",
    "Table with various cookies, donuts, cakes, coffee cups on pink background painting by Van Gogh",
    "A beautiful teen girl with long hair and a hair bun, a flower in her hair, gentle smile, blue eyes, a character portrait, pre-raphaelitism, painting by Van Gogh, enchanting",
    "Cinematic shot of (taxidermy bird:1.2) inside a glass ball painting by Van Gogh, antique shop, glass, crystal, flowers, moss, rocks, luxurious, terrarium",
    "Inside a Swiss chalet, snow capped mountain view out of window, with a fireplace, at night, interior design, d & d concept art, d & d wallpaper, warm, art painting by Van Gogh, extremely detailed, intricate, beautiful, colourful.",
    "A beautiful woman, painting by Van Gogh",
    "The face of happiness by Van Gogh"
    "A beautiful wood building with a river side street street. Beautiful in look at its own made-in-air artwork by Van Gogh"
]

for unet_path in unet_paths:
    unet = UNet2DConditionModel.from_pretrained(unet_path)

    checkpoint_name = unet_path.split('/')[-2]  # Extracts 'checkpoint-xxxxx'
    model_suffix = checkpoint_name.split('-')[-1]  # Extracts 'xxxxx'

    out_dir = base_out_dir + model_suffix + "/"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path=model_id, safety_checker=None, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path=model_id, safety_checker=None,
                                                                  torch_dtype=torch.float16)
    pipe_img2img = pipe_img2img.to("cuda")

    pipeline = DiffusionPipeline.from_pretrained(model_id, unet=unet, dtype=torch.float16)
    pipeline = pipeline.to("cuda")

    print(f"Model is on: {pipeline.device}")

    pipeline.enable_xformers_memory_efficient_attention()

    for main_prompt in prompts:
        negative = "(worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), poorly drawn, low resolution, ugly, nude, tiling, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"

        prompt = "(van_gogh) " + main_prompt
        short_prompt = prompt[0:40]

        image = pipeline(prompt, guidance_scale=8, negative_prompt=negative).images[0]
        image.save(out_dir + short_prompt  + "_512.png")

        image = image.resize((768, 768), resample=Image.BOX)
        image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.65, negative_prompt=negative).images[0]
        image.save(out_dir + short_prompt + "_768.png")

torch.cuda.empty_cache()