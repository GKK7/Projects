import argparse
import os
import torch
import subprocess
from pytorch_lightning import seed_everything
from pathlib import Path
import time
import glob
from termcolor import colored
import warnings
from pkg_resources import PkgResourcesDeprecationWarning
import logging
import sys

logging.basicConfig(level=logging.CRITICAL)

warnings.filterwarnings("ignore", category=PkgResourcesDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore",
                        message=".*diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline*")


def count_images_in_folder(image_folder):
    try:
        jpg_count = len(glob.glob(os.path.join(image_folder, "*.jpg")))
        png_count = len(glob.glob(os.path.join(image_folder, "*.png")))
        total_images = jpg_count + png_count
        return total_images
    except Exception as e:
        print(f"Error while counting images in {image_folder}: {e}")
        return 0


def parse_arguments():
    current_folder = os.getcwd()
    storage_folder = os.path.join(current_folder, "../storage/dreambooth_automation")
    os.makedirs(storage_folder, exist_ok=True)

    parser = argparse.ArgumentParser()

    pretrained_model_folder = os.path.join(storage_folder, "pretrained_model/")
    parser.add_argument("--pretrained_model_path", default=pretrained_model_folder,
                        type=str, help="Path to the pretrained model.")

    output_models_folder = os.path.join(storage_folder, "generated_models/")
    os.makedirs(output_models_folder, exist_ok=True)
    parser.add_argument("--output_dir", default=output_models_folder, type=str,
                        help="Directory for output.")

    image_folder_default = os.path.join(storage_folder, "test_images/")
    parser.add_argument("--image_folder", default=image_folder_default, type=str,
                        help="Base folder containing subfolders.")

    logs_folder = os.path.join(storage_folder, "Logs/")
    os.makedirs(logs_folder, exist_ok=True)
    parser.add_argument("--log_dir", default=logs_folder, type=str,
                        help="Directory for logs.")

    parser.add_argument("--steps_per_example", type=int, default=35, help="Number of steps per example image.")
    parser.add_argument("--max_train_steps_list", nargs='+', type=int, default=None,
                        help="List of max training steps.")

    parser.add_argument("--batch_size_list", nargs='+', type=int, default=[1],
                        help="List of batch sizes.")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution of the output.")

    args = parser.parse_args()

    return args


args = parse_arguments()

torch.cuda.set_per_process_memory_fraction(0.8)

# Check CUDA availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('============================== CUDA INFO ==============================')
    print('Total GPU Memory:', torch.cuda.get_device_properties(device).total_memory)
    print('Allocated GPU Memory:', torch.cuda.memory_allocated(device))
    print('Cached GPU Memory:', torch.cuda.memory_reserved(device))
    print('======================================================================')
else:
    print('============================== CUDA INFO ==============================')
    print('CUDA is not available.')
    print('======================================================================')

torch.cuda.empty_cache()

log_dir = args.log_dir


def train_model(max_train_steps, learning_rate, batch_size, unique_output_dir):
    # Check if the output directory already exists
    if os.path.exists(unique_output_dir) and os.path.exists(os.path.join(unique_output_dir, "model_index.json")):
        print(colored(f"Model {unique_output_dir} already exists. Skipping...\n", "magenta"))
        return

    current_folder = os.getcwd()
    vendor_script = os.path.expanduser(current_folder + "/../vendor/train_dreambooth.py")
    accelerate_command = [
        "accelerate",
        "launch",
        vendor_script,
        "--enable_xformers_memory_efficient_attention",
        f"--pretrained_model_name_or_path={args.pretrained_model_path}",
        f"--instance_data_dir={instance_data_dir}",
        f"--output_dir={unique_output_dir}",
        f"--instance_prompt={instance_prompt}",
        f"--resolution={resolution}",
        f"--train_batch_size={batch_size}",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        f"--learning_rate={learning_rate}",
        f"--lr_scheduler={lr_scheduler}",
        f"--lr_warmup_steps={lr_warmup_steps}",
        f"--max_train_steps={max_train_steps}",
        f"--checkpointing_steps={checkpoint_steps}"
    ]

    process = subprocess.Popen(accelerate_command, stdout=sys.stdout, stderr=sys.stderr)
    process.communicate()

    if process.returncode == 0:
        print(colored("Training completed successfully.\n", "green"))
    else:
        print(colored("Error executing command:\n", "red"))
        #print(stderr.decode())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


output_dir = args.output_dir
resolution = args.resolution
batch_size_list = args.batch_size_list
gradient_accumulation_steps = 1
learning_rate_list = [4e-6]
lr_scheduler = "constant"
lr_warmup_steps = 0
max_train_steps_list = args.max_train_steps_list
checkpoint_steps = 10000

seed_everything(123)
torch.cuda.empty_cache()

# Base folder containing subfolders
image_folder = args.image_folder

# Get all subfolders in the image_folder
subfolders = [f.name for f in os.scandir(image_folder) if f.is_dir()]

# Loop over each subfolder
for subfolder in subfolders:
    # Define the path to the model_index.json file in the subfolder
    model_index_file = os.path.join(args.image_folder, subfolder, 'model_index.json')

    # If model_index.json exists in the subfolder, skip this folder
    if os.path.exists(model_index_file):
        print(f"'model_index.json' file exists in {subfolder}. Skipping...\n")
        continue

    instance_prompt = f"_{subfolder}_"
    instance_data_dir = os.path.join(args.image_folder, subfolder)

    # If max_train_steps_list is not provided, calculate it based on the image count
    max_train_steps_list = 0
    ignore_steps_per_example = False
    if args.max_train_steps_list is None:
        image_count = count_images_in_folder(instance_data_dir)
        max_train_steps_list = [image_count * args.steps_per_example]
    else:
        max_train_steps_list = args.max_train_steps_list
        ignore_steps_per_example = True

    # Check if the subfolder is empty
    if not os.listdir(instance_data_dir):
        print(f"Subfolder {subfolder} is empty. Skipping...\n")
        continue

    image_count = count_images_in_folder(instance_data_dir)

    print(colored(f"Dreambooth start training folder: " + subfolder, "yellow"))
    print("    image_count: " + colored(image_count, "yellow"))
    if (ignore_steps_per_example):
        print("    steps_per_example: " + colored(f"{args.steps_per_example} (Ignored)", "yellow"))
    else:
        print("    steps_per_example: " + colored(args.steps_per_example, "yellow"))
    print("    max_train_steps_list: " + colored(max_train_steps_list, "yellow"))
    print()

    for lr in learning_rate_list:
        for batch_size in batch_size_list:
            for steps in max_train_steps_list:
                torch.cuda.empty_cache()

                # unique output directory for each run
                # check if batch_size is 1
                if batch_size == 1:
                    unique_output_dir = f"{output_dir}_{subfolder}_{Path(args.pretrained_model_path).stem}_steps_{steps}"
                else:
                    unique_output_dir = f"{output_dir}_{subfolder}_{Path(args.pretrained_model_path).stem}_steps_{steps}_batch_{batch_size}"

                start_time = time.time()
                train_model(steps, lr, batch_size, unique_output_dir)
                end_time = time.time()
                training_time = end_time - start_time
                print("Time for training with max_train_steps =", steps, ", learning_rate =", lr, ", batch_size =",
                      batch_size, ":", training_time, "seconds")

                with open(f"{log_dir}/training_logs.txt", "a") as log_file:
                    log_file.write(
                        f"Model: {unique_output_dir}, Time for training with max_train_steps = {steps}, learning_rate = {lr}, batch_size = {batch_size}: {training_time} seconds\n")
