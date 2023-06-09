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


log_dir = "/home/gkirilov/Brand_new_PC/Logs"


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


instance_data_dir = "/home/gkirilov/Checkpoint/markle_harry"
output_dir = "/home/gkirilov/Brand_new_PC/markleharry/dreambooth"
instance_prompt = "photo of markleharry"
resolution = 512
batch_size = 1
gradient_accumulation_steps = 1
learning_rate_list = [4e-6]
lr_scheduler = "constant"
lr_warmup_steps = 0
max_train_steps_list = [600, 800, 1000, 1300, 1600, 2000, 2500, 3000]


seed_everything(123)
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

        with open(f"{log_dir}log_mickey.txt", "a") as log_file:
            log_file.write(
                f"Time for training with max_train_steps = {steps}, learning_rate = {lr}, batch_size = {batch_size}: {training_time} seconds\n")


torch.cuda.empty_cache()


