import json
import os
from termcolor import colored

def get_directories():
    current_folder = os.getcwd()
    storage_folder = os.path.join(current_folder, "../storage/dreambooth_automation")
    os.makedirs(storage_folder, exist_ok=True)
    generated_prompts_folder = os.path.join(storage_folder, "generated_prompts/json_results/")
    os.makedirs(generated_prompts_folder, exist_ok=True)
    processed_dir = os.path.join(generated_prompts_folder, 'processed_results')
    os.makedirs(processed_dir, exist_ok=True)
    return generated_prompts_folder, processed_dir

def process_file(file, generated_prompts_folder, processed_dir):
    processed_file_path = os.path.join(processed_dir, file.split('.')[0] + '_processed_results.txt')
    if os.path.exists(processed_file_path):
        print(colored(f"Processed file for {file} already exists. Skipping...\n", "magenta"))
        return False

    with open(os.path.join(generated_prompts_folder, file), 'r') as json_file:
        data = json.load(json_file)

    prompts = [item["text"] for item in data]
    generated_results = [prompt for prompt in prompts]

    with open(processed_file_path, 'w') as result_file:
        result_file.write('\n'.join(generated_results))

    return True

def main():
    generated_prompts_folder, processed_dir = get_directories()

    files = os.listdir(generated_prompts_folder)
    processed_any = False

    for file in files:
        if file != 'processed_results':
            processed = process_file(file, generated_prompts_folder, processed_dir)
            if processed:
                processed_any = True

    if processed_any:
        print(colored("\nProcessed BLIP captions and created new files successfully.", "green"))

if __name__ == "__main__":
    main()
