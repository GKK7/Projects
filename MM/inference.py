import argparse
import os
import sys
import glob
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from pathlib import Path
from PIL import Image
import warnings
from termcolor import colored

warnings.filterwarnings("ignore")


def infer_models(model_dir, generated_images_dir):
    Path(generated_images_dir).mkdir(parents=True, exist_ok=True)

    model_name = os.path.basename(model_dir)
    out_dir = f"{generated_images_dir}/{model_name}/"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(model_dir, safety_checker=None, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(model_dir, safety_checker=None,
                                                                  torch_dtype=torch.float16)
    pipe_img2img = pipe_img2img.to("cuda")

    print(colored(f"Model {model_name} is on: {pipe.device}", "yellow"))
    print(colored(f"Image-to-Image model {model_name} is on: {pipe_img2img.device}", "yellow"))

    pipe.enable_xformers_memory_efficient_attention()
    pipe_img2img.enable_xformers_memory_efficient_attention()

    # Load instance prompts from the file
    current_folder = os.getcwd()
    storage_folder = os.path.join(current_folder, "../storage/dreambooth_automation")
    with open(storage_folder + '/instance_prompt.txt', 'r') as f:
        instance_prompts = f.read().strip().split('\n')

    # Get the first non-empty part of the model name (splitting on underscores)
    model_parts = [part for part in model_name.split('_') if part]
    model_prefix = model_parts[0] if model_parts else ''

    # Check if there is a corresponding prompt file for this model
    prompt_file_path = None
    for file_name in os.listdir(os.path.join(storage_folder, 'generated_prompts/gpt_generated_prompts')):
        if file_name.startswith(model_prefix):
            prompt_file_path = os.path.join(storage_folder, 'generated_prompts/gpt_generated_prompts', file_name)
            break

    if not prompt_file_path:
        print(colored(f"\nNo prompt file found for model {model_name}. Skipping.", "magenta"))
        return

    # Load prompts and negatives from the corresponding file
    with open(prompt_file_path, 'r') as f:
        # This splits the file content on two consecutive newlines,
        # which is the separator between different prompts and negatives
        blocks = f.read().strip().split('\n\n')
        prompts_and_negatives = []
        for block in blocks:
            lines = block.split('\n')
            # Make sure there are at least 3 lines in the block
            if len(lines) >= 3:
                # We are skipping the subject, only appending the prompts and negatives
                prompts_and_negatives.append({'prompt': lines[1][8:], 'negative': lines[2][10:]})
            else:
                print(f"Skipping block due to insufficient lines: {block}")

    for instance_prompt in instance_prompts:
        if instance_prompt.lower() in model_name.lower():  # Case-insensitive match
            for prompt_and_negative in prompts_and_negatives:
                prompt = f"{instance_prompt} {prompt_and_negative['prompt']}"
                filename_base = prompt[0:50]
                negative = prompt_and_negative['negative']

                # Skip if the 1024 version already exists
                filepath_1024 = Path(out_dir, filename_base + "_1024.png")
                if filepath_1024.is_file():
                    temp_filename = filename_base + "_1024.png"
                    print(colored(f"File '{temp_filename}' already exists. Skipping.", "magenta"))
                    continue

                print()
                print(colored(f"Prompt: {prompt}", "yellow"))

                # 512
                image = pipe(prompt, guidance_scale=8, negative_prompt=negative).images[0]
                image.save(Path(out_dir, filename_base + "_512.png"))

                # 768
                image = image.resize((768, 768), resample=Image.BOX)
                image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.65,
                                     negative_prompt=negative).images[0]
                image.save(Path(out_dir, filename_base + "_768.png"))

                # 1024
                image = image.resize((1024, 1024), resample=Image.BOX)
                image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.55,
                                     negative_prompt=negative).images[0]
                image.save(Path(out_dir, filename_base + "_1024.png"))

                print()

                torch.cuda.empty_cache()


if __name__ == '__main__':

    current_folder = os.getcwd()
    storage_folder = os.path.join(current_folder, "../storage/dreambooth_automation")
    os.makedirs(storage_folder, exist_ok=True)

    parser = argparse.ArgumentParser()

    models_dir = os.path.join(storage_folder, "generated_models/")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    parser.add_argument("--models_dir", default=models_dir,
                        type=str, help="Directory where the models are located.")

    generated_images_dir = os.path.join(storage_folder, "generated_images/")
    if not os.path.exists(generated_images_dir):
        os.makedirs(generated_images_dir)
    parser.add_argument("--generated_images_dir", default=generated_images_dir,
                        type=str, help="Output directory where the generated images are saved.")

    args = parser.parse_args()

    main_dir = args.models_dir
    output_base_dir = args.generated_images_dir
    for sub_dir in glob.glob(os.path.join(main_dir, '*')):
        infer_models(sub_dir, output_base_dir)
