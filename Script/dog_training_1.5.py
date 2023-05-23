import subprocess
import torch

torch.cuda.set_per_process_memory_fraction(0.8)

accelerate_command = [
    "accelerate",
    "launch",
    "train_dreambooth.py",
    "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
    "--instance_data_dir=/home/gkirilov/alvan",
    "--class_data_dir=/home/gkirilov/PycharmProjects/models_manager/path-to-class-/home/gkirilov/Realistic_jenna",
    "--output_dir=path-to-save-model",
    "--with_prior_preservation",
    "--prior_loss_weight=1.0",
    "--instance_prompt=a photo of alvanx dog",
    "--class_prompt=a photo of dog",
    "--resolution=512",
    "--train_batch_size=1",
    "--gradient_accumulation_steps=1",
    "--learning_rate=5e-6",
    "--lr_scheduler=constant",
    "--lr_warmup_steps=0",
    "--num_class_images=200",
    "--max_train_steps=1000"
]

process = subprocess.Popen(accelerate_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

if process.returncode == 0:
    print("Command executed successfully.")
else:
    print("Error executing command:")
    print(stderr.decode())
