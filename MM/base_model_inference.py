import argparse
import os
import glob
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from pathlib import Path
from PIL import Image
import warnings
from termcolor import colored

warnings.filterwarnings("ignore")

def infer_model(model_path, generated_images_dir, prompt_dir):
    Path(generated_images_dir).mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(model_path, safety_checker=None,
                                                                  torch_dtype=torch.float16)
    pipe_img2img = pipe_img2img.to("cuda")

    print(colored(f"Model is on: {pipe.device}", "yellow"))
    print(colored(f"Image-to-Image model is on: {pipe_img2img.device}", "yellow"))

    pipe.enable_xformers_memory_efficient_attention()
    pipe_img2img.enable_xformers_memory_efficient_attention()

    # Process all .txt files in the gpt_generated_prompts directory
    for prompt_file_path in glob.glob(prompt_dir + '/*.txt'):
        print(colored(f"\nProcessing prompt file {prompt_file_path}", "yellow"))

        # Create subdirectory for each .txt file
        txt_filename = os.path.basename(prompt_file_path).replace(".txt", "")
        current_generated_images_dir = os.path.join(generated_images_dir, "base_model_" + txt_filename)
        os.makedirs(current_generated_images_dir, exist_ok=True)

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

        for prompt_and_negative in prompts_and_negatives:
            prompt = prompt_and_negative['prompt']
            filename_base = prompt[0:50]
            negative = prompt_and_negative['negative']

            # Skip if the 1024 version already exists
            filepath_1024 = Path(current_generated_images_dir, filename_base + "_1024.png")
            if filepath_1024.is_file():
                temp_filename = filename_base + "_1024.png"
                print(colored(f"File '{temp_filename}' already exists. Skipping.", "magenta"))
                continue

            print()
            print(colored(f"Prompt: {prompt}", "yellow"))

            # 512
            image = pipe(prompt, guidance_scale=8, negative_prompt=negative).images[0]
            image.save(Path(current_generated_images_dir, filename_base + "_512.png"))

            # 768
            image = image.resize((768, 768), resample=Image.BOX)
            image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.65,
                                 negative_prompt=negative).images[0]
            image.save(Path(current_generated_images_dir, filename_base + "_768.png"))

            # 1024
            image = image.resize((1024, 1024), resample=Image.BOX)
            image = pipe_img2img(prompt=prompt, guidance_scale=8, image=image, strength=0.55,
                                 negative_prompt=negative).images[0]
            image.save(Path(current_generated_images_dir, filename_base + "_1024.png"))

            print()

            torch.cuda.empty_cache()


if __name__ == '__main__':

    current_folder = os.getcwd()
    storage_folder = os.path.join(current_folder, "../storage/dreambooth_automation")
    os.makedirs(storage_folder, exist_ok=True)

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, help="Path to the trained model.")
    generated_images_dir = os.path.join(storage_folder, "generated_images/")
    if not os.path.exists(generated_images_dir):
        os.makedirs(generated_images_dir)
    parser.add_argument("--generated_images_dir", default=generated_images_dir,
                        type=str, help="Output directory where the generated images are saved.")

    gpt_generated_prompts_dir = os.path.join(storage_folder, "generated_prompts/gpt_generated_prompts")
    if not os.path.exists(gpt_generated_prompts_dir):
        os.makedirs(gpt_generated_prompts_dir)
    parser.add_argument("--prompt_dir", default=gpt_generated_prompts_dir,
                        type=str, help="Directory where the GPT-generated prompts are located.")

    args = parser.parse_args()

    infer_model(args.model_path, args.generated_images_dir, args.prompt_dir)
