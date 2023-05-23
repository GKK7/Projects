import os
import subprocess

MODEL_NAME = "CompVis/stable-diffusion-v1-4"
INSTANCE_DIR = "/home/gkirilov/Jenna_Ortega_base"
OUTPUT_DIR = "/home/gkirilov/Jenna_Ortega_base/dreambooth"

# Set the environment variables
os.environ["MODEL_NAME"] = MODEL_NAME
os.environ["INSTANCE_DIR"] = INSTANCE_DIR
os.environ["OUTPUT_DIR"] = OUTPUT_DIR

# Command to execute
command = [
    "accelerate",
    "launch",
    "train_dreambooth.py",
    "--pretrained_model_name_or_path", MODEL_NAME,
    "--instance_data_dir", INSTANCE_DIR,
    "--output_dir", OUTPUT_DIR,
    "--instance_prompt", "a photo of sks dog",
    "--resolution", "512",
    "--train_batch_size", "1",
    "--gradient_accumulation_steps", "1",
    "--learning_rate", "5e-6",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--max_train_steps", "400",
    "--push_to_hub"
]

# Run the command
subprocess.run(command, capture_output=True)