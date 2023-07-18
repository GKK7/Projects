import openai
import os
import time
import random
from termcolor import colored
from tqdm import tqdm

# Load the API key
OPENAI_API_KEY = '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'
openai.api_key = OPENAI_API_KEY

current_folder = os.getcwd()
storage_folder = os.path.join(current_folder, "../storage/dreambooth_automation")
processed_results_folder = os.path.join(storage_folder, "generated_prompts/json_results/processed_results/")
gpt_generated_prompts_folder = os.path.join(storage_folder, "generated_prompts/gpt_generated_prompts")
os.makedirs(gpt_generated_prompts_folder, exist_ok=True)

with open(current_folder + '/../dreambooth_automation_3/prompt_instructions.txt', 'r') as file:
    instructions = file.read().replace('\n', ' ')

MAX_RETRIES = 10
BACKOFF_FACTOR = 0.5


def get_files_to_process():
    files = [f for f in os.listdir(processed_results_folder) if os.path.isfile(os.path.join(processed_results_folder, f))]
    return [filename for filename in files if not os.path.exists(os.path.join(gpt_generated_prompts_folder, filename.rsplit('.', 1)[0] + '_gpt_generated_prompts.txt'))]


def get_response_from_api(messages):
    for attempt in range(MAX_RETRIES):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                max_tokens=150,
                temperature=0.5,
                messages=messages)
            return response
        except (openai.error.ServiceUnavailableError, openai.error.APIError) as e:
            if attempt == MAX_RETRIES - 1:
                print(colored(f"\nFailed on final attempt with error: {e}", "red"))
                raise
            else:
                wait_time = BACKOFF_FACTOR * (2 ** attempt)
                print(colored(f"\nAttempt {attempt + 1} failed. Retrying after {wait_time:.2f} seconds...", "yellow"))
                time.sleep(wait_time + random.random())


def truncate_string(s, max_len):
    if len(s) > max_len:
        comma_index = s.find(',', max_len)
        return s[:comma_index] if comma_index != -1 else s
    return s


def process_files():
    files_to_process = get_files_to_process()

    if not files_to_process:
        print(colored("\nAll files have already been processed. Skipping...", "magenta"))
        return

    for filename in tqdm(files_to_process, desc="Processing directories"):
        output_file_name = filename.rsplit('.', 1)[0] + '_gpt_generated_prompts.txt'
        output_file_path = os.path.join(gpt_generated_prompts_folder, output_file_name)

        with open(os.path.join(processed_results_folder, filename), 'r') as file:
            subjects = file.read().splitlines()

        max_subjects = 30  # Maximum number of subjects to process
        subjects = subjects[:max_subjects]

        with open(output_file_path, 'w') as output_file:
            for subject in tqdm(subjects, desc="Generating GPT prompts", leave=False):
                messages = [{"role": "system", "content": instructions}, {"role": "user", "content": subject}]
                response = get_response_from_api(messages)
                ai_message = response['choices'][0]['message']['content']

                split_message = ai_message.split('NEGATIVE:', 1)
                description_part = truncate_string(split_message[0], 250)
                negative_part = truncate_string(split_message[1][:125] if len(split_message) > 1 else '', 125)

                output_file.write(f"Subject: {subject}\nPrompt: {description_part}\n")

                if negative_part:
                    output_file.write(f"NEGATIVE: {negative_part}\n")
                else:
                    messages.append({"role": "user", "content": "What are the potential negative aspects of this scene?"})
                    response = get_response_from_api(messages)
                    ai_message = response['choices'][0]['message']['content']
                    output_file.write(f"{ai_message[:150]}\n")

                output_file.write("\n")

    print(colored("\nGPT prompt files created successfully.", "green"))


try:
    process_files()
except Exception as e:
    print(colored("\nAn error occurred:", str(e), "red"))
