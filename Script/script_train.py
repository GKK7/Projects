import subprocess
import torch

torch.cuda.set_per_process_memory_fraction(0.8)

accelerate_command = [
    "accelerate",
    "launch",
    "train_dreambooth.py",
    "--pretrained_model_name_or_path=SG161222/Realistic_Vision_V1.4",  # Replace "model_name" with the actual model name
    "--instance_data_dir=/home/gkirilov/Jenna_Ortega_base",  # Replace "instance_dir" with the actual instance data directory
    "--output_dir=/home/gkirilov/Jenna_Ortega_base",  # Replace "output_dir" with the actual output directory
    "--instance_prompt=a photo jennaortegax",
    "--resolution=512",
    "--train_batch_size=1",
    "--gradient_accumulation_steps=1",
    "--learning_rate=5e-6",
    "--lr_scheduler=constant",
    "--lr_warmup_steps=0",
    "--max_train_steps=600"
]

process = subprocess.Popen(accelerate_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

if process.returncode == 0:
    print("Command executed successfully.")
else:
    print("Error executing command:")
    print(stderr.decode())
