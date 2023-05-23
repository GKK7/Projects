import torch
from transformers import AutoModel, AutoTokenizer

# Instantiate the model
model = AutoModel.from_pretrained('/home/gkirilov/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/aa9ba505e1973ae5cd05f5aedd345178f52f8e6a')

# Load the weights
model.load_state_dict(torch.load('/home/gkirilov/Documents/LORAs/diffusers/examples/dreambooth'))

tokenizer = AutoTokenizer.from_pretrained('/home/gkirilov/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/aa9ba505e1973ae5cd05f5aedd345178f52f8e6a')

input_text = "Your text here"
encoded_input = tokenizer(input_text, return_tensors='pt')

with torch.no_grad():
    model_output = model(**encoded_input)

model.eval()