import os
import subprocess

MODEL_NAME = "/home/gkirilov/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/aa9ba505e1973ae5cd05f5aedd345178f52f8e6a"
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
    "/home/gkirilov/Documents/LORAs/diffusers/examples/dreambooth/train_dreambooth.py",
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
    "--max_train_steps", "400"
]

# Run the command
subprocess.run(command, capture_output=True)