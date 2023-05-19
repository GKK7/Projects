import os
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
    captions = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(pixel_values, **gen_kwargs)

        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        caption = preds[0].strip()
        captions.append(caption)

        # Save the caption as a text file with the same name as the original image
        image_filename = os.path.basename(image_path)
        caption_filename = os.path.splitext(image_filename)[0] + ".txt"
        caption_path = os.path.join("/home/gkirilov/Jenna_Ortega_resized", caption_filename)
        with open(caption_path, "w") as f:
            f.write(caption)

    return captions


image_folder = "/home/gkirilov/Jenna_Ortega_resized"
image_files = os.listdir(image_folder)
image_paths = [os.path.join(image_folder, filename) for filename in image_files if filename.endswith(".jpg")]

generated_captions = predict_step(image_paths)

for caption, image_path in zip(generated_captions, image_paths):
    print(f"Image: {image_path}")
    print(f"Caption: {caption}")
    print()
