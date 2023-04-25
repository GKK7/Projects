import torch
print(torch.cuda.device_count(), 'GPUs available')
print('Current device:', torch.cuda.current_device())
print('Device name:', torch.cuda.get_device_name(torch.cuda.current_device()))

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
print(tokens)