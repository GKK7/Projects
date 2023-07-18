import os
import argparse
from termcolor import colored

# Define the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--images_folder", required=True, help="Path to folder containing images")
parser.add_argument("--pretrained_model_path", required=True, help="Path to the pretrained model")
parser.add_argument("--steps_per_example", default=35, type=int,
                    help="Steps per example, defaults to 35x number of images")
parser.add_argument("--max_train_steps_list", nargs='+', type=int,
                    help="Overrides '--steps_per_example'. Specify the number of training steps. You can give different, e.g. 200 400 500.")

# Parse the arguments
args = parser.parse_args()

# Training script parameters
train_script_name = "training.py"
pretrained_model_path = args.pretrained_model_path  # path to cached model
image_folder = args.images_folder  # path to folder containing subfolders of images
resolution = 512
batch_size_list = [1]

# Get all subfolders in the image_folder
subfolders = [f.name for f in os.scandir(image_folder) if f.is_dir()]

# Write the instance prompts to file
current_folder = os.getcwd()
storage_folder = os.path.join(current_folder, "../storage/dreambooth_automation")
os.makedirs(storage_folder, exist_ok=True)
with open(storage_folder + '/instance_prompt.txt', 'w') as f:
    for subfolder in subfolders:
        instance_prompt = f"_{subfolder}_"
        f.write(instance_prompt + '\n')

batch_size_list_str = ' '.join(map(str, batch_size_list))

# Call the training script parameters
training_command = (
    f"python3 {train_script_name} "
    f"--pretrained_model_path {pretrained_model_path} "
    f"--resolution {resolution} "
    f"--batch_size_list {batch_size_list_str} "
    f"--image_folder {image_folder}"
)

if args.max_train_steps_list is not None:
    max_train_steps_str = ' '.join(map(str, args.max_train_steps_list))
    training_command = training_command + f" --max_train_steps_list {max_train_steps_str}"

if args.steps_per_example is not None:
    training_command = training_command + f" --steps_per_example {args.steps_per_example}"

# Run inference and image grid scripts
infer_script_name = "inference.py"
grid_script_name = "image_grid.py"
generate_prompts_script = "generate_captions.py"
blip_to_prompts = "blip_to_prompts.py"
gpt_prompts = "gpt_prompting.py"
base_model_inference = "base_model_inference.py"

os.system(training_command)
os.system(f"python3 {generate_prompts_script} --images_folder {image_folder}")
os.system(f"python3 {blip_to_prompts}")
os.system(f"python3 {gpt_prompts}")
os.system(f"python3 {infer_script_name}")
os.system(f"python3 {base_model_inference} --model_path {pretrained_model_path}")
os.system(f"python3 {grid_script_name}")

# Delete the instance_prompt.txt file
if os.path.exists(storage_folder + '/instance_prompt.txt'):
    os.remove(storage_folder + '/instance_prompt.txt')
else:
    print("The file does not exist")
