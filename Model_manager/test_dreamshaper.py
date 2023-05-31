import subprocess
import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from pytorch_lightning import seed_everything
from pathlib import Path
from PIL import Image
from train_dreambooth import parse_args
from train_dreambooth import DreamBoothDataset, tokenize_prompt
import time
from datetime import datetime

torch.cuda.set_per_process_memory_fraction(0.8)

# Check CUDA availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Total GPU Memory:', torch.cuda.get_device_properties(device).total_memory)
    print('Allocated GPU Memory:', torch.cuda.memory_allocated(device))
    print('Cached GPU Memory:', torch.cuda.memory_reserved(device))
else:
    print('CUDA is not available.')

torch.cuda.empty_cache()

log_dir = "/home/gkirilov/Brand_new_PC/Logs/"


def train_model(max_train_steps, learning_rate, batch_size, unique_output_dir):
    accelerate_command = [
        "accelerate",
        "launch",
        "train_dreambooth.py",
        f"--pretrained_model_name_or_path={pretrained_model}",
        f"--instance_data_dir={instance_data_dir}",
        f"--output_dir={unique_output_dir}",
        f"--instance_prompt={instance_prompt}",
        f"--resolution={resolution}",
        f"--train_batch_size={batch_size}",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        f"--learning_rate={learning_rate}",
        f"--lr_scheduler={lr_scheduler}",
        f"--lr_warmup_steps={lr_warmup_steps}",
        f"--max_train_steps={max_train_steps}"
    ]

    process = subprocess.Popen(accelerate_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        print("Command executed successfully.")
    else:
        print("Error executing command:")
        print(stderr.decode())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


pretrained_model = "/home/gkirilov/models/Dreamshaper/dreamshaper_6BakedVae"

instance_data_dir = "/home/gkirilov/Brand_new_PC/stash"
output_dir = "/home/gkirilov/Brand_new_PC/dreambooth_raw/dreambooth"
instance_prompt = "a photo of JENNAORTEGAX"
resolution = 512
batch_size = 1
gradient_accumulation_steps = 1
learning_rate_list = [4e-6, 3e-6]
lr_scheduler = "constant"
lr_warmup_steps = 0
max_train_steps_list = [400, 600, 800, 1000, 1200]

seed_everything(123)
torch.cuda.empty_cache()


def test_models(max_train_steps, learning_rate, pretrained_model):
    models = {
        "RV1.4": {
            "path": pretrained_model,
            "active": True,
            "keyword": "JENNAORTEGAX_solo_raw"
        },
    }

    for model_name, model_info in models.items():
        if model_info["active"]:
            pipe = StableDiffusionPipeline.from_pretrained(model_info["path"], safety_checker=None,
                                                           torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(model_info["path"], safety_checker=None,
                                                                          torch_dtype=torch.float16)
            pipe_img2img = pipe_img2img.to("cuda")

            print(f"Model {model_name} is on: {pipe.device}")
            print(f"Image-to-Image model {model_name} is on: {pipe_img2img.device}")

            pipe.enable_xformers_memory_efficient_attention()
            pipe_img2img.enable_xformers_memory_efficient_attention()

            out_dir = f"/home/gkirilov/Checkpoint/{model_info['keyword']}/steps_{max_train_steps}_lr_{learning_rate}_batch_{batch_size}/"
            Path(out_dir).mkdir(parents=True, exist_ok=True)

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
                negative = "(worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), poorly drawn, low resolution, ugly, nude, tiling, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy, artist logo"

                prompt = "(" + model_info["keyword"] + ") " + main_prompt
                short_prompt = prompt[0:40]

                image = pipe(prompt, guidance_scale=8, negative_prompt=negative).images[0]
                image.save(out_dir + short_prompt + "_JORTEGA100_512.png")

                image = image.resize((768, 768), resample=Image.BOX)
                image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.65,
                                     negative_prompt=negative).images[0]
                image.save(out_dir + short_prompt + "_JORTEGA100_768.png")

                image = image.resize((1024, 1024), resample=Image.BOX)
                image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.55,
                                     negative_prompt=negative).images[0]
                image.save(out_dir + short_prompt + "_JORTEGA100_1024.png")

                # image = image.resize((1280, 1280), resample=Image.BOX)
                # image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.45,
                #                      negative_prompt=negative).images[0]
                # image.save(out_dir + short_prompt + "_JORTEGA1280.png")


torch.cuda.empty_cache()

for lr in learning_rate_list:
    for steps in max_train_steps_list:
        # unique output directory for each run
        unique_output_dir = f"{output_dir}_{Path(pretrained_model).stem}_steps_{steps}_lr_{lr}_batch_{batch_size}"  # Update the line

        start_time = time.time()
        train_model(steps, lr, batch_size, unique_output_dir)
        end_time = time.time()
        training_time = end_time - start_time
        print("Time for training with max_train_steps =", steps, ", learning_rate =", lr, ", batch_size =",
              batch_size, ":", training_time, "seconds")

        with open(f"{log_dir}log_dream_raw.txt", "a") as log_file:
            log_file.write(
                f"Time for training with max_train_steps = {steps}, learning_rate = {lr}, batch_size = {batch_size}: {training_time} seconds\n")

        start_time = time.time()
        test_models(steps, lr, pretrained_model)  # pass steps and pretrained_model as arguments to test_models
        end_time = time.time()
        test_models_time = end_time - start_time
        print("Time for test_models with max_train_steps =", steps, ":", test_models_time, "seconds")

        # Log test_models time
        with open(f"{log_dir}log_dream_raw.txt", "a") as log_file:
            log_file.write(f"Time for test_models with max_train_steps = {steps}: {test_models_time} seconds\n")

        torch.cuda.empty_cache()

torch.cuda.empty_cache()
