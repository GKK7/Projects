import os
import subprocess

# Constants
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
INSTANCE_DIR = "/home/gkirilov/Jenna_Ortega_base"
OUTPUT_DIR = "/home/gkirilov/Jenna_Ortega_resized"
SCRIPT_PATH = "/home/gkirilov/Jenna_Ortega_resized/diffusers/examples/dreambooth/train_dreambooth_lora.py"

# Install required packages
subprocess.run(["pip", "install", "wandb"])
subprocess.run(["pip", "install", "torchvision"])
subprocess.run(["pip", "install", "torch==2.0.1"])

# Accelerate launch command
subprocess.run(["accelerate", "launch", SCRIPT_PATH,
  "--pretrained_model_name_or_path=" + MODEL_NAME,
  "--instance_data_dir=" + INSTANCE_DIR,
  "--output_dir=" + OUTPUT_DIR,
  "--instance_prompt='a photo of sks dog'",
  "--resolution=512",
  "--train_batch_size=1",
  "--gradient_accumulation_steps=1",
  "--checkpointing_steps=100",
  "--learning_rate=1e-4",
  "--report_to='wandb'",
  "--lr_scheduler='constant'",
  "--lr_warmup_steps=0",
  "--max_train_steps=500",
  "--validation_prompt='A photo of woman with black hair and dark eyes'",
  "--validation_epochs=50",
  "--seed='0'"])
