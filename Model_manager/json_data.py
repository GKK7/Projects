import os
import json

# Step 1: Define the directory path
directory = "/home/gkirilov/stable-diffusion/LORA_training_images/100/Image/100_test"

# Step 3: Create a Python list
data_list = []

# Step 4: Iterate through images and captions in the directory
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        image_path = os.path.join(directory, filename)
        caption_path = os.path.join(directory, filename.split(".")[0] + ".txt")

        with open(caption_path, "r") as caption_file:
            caption_text = caption_file.read()

        image_obj = {
            "image_path": image_path,
            "caption": caption_text
        }
        data_list.append(image_obj)

# Step 6: Convert List to JSON and Save to File
json_data = json.dumps(data_list, indent=4)  # Convert list to JSON string with indentation
file_path = "/home/gkirilov/stable-diffusion/LORA_training_images/100/Image/100_test/image_data.json"  # Specify the file path where you want to save the JSON file

with open(file_path, "w") as json_file:
    json_file.write(json_data)

print("Image data and captions converted and saved to JSON file successfully.")