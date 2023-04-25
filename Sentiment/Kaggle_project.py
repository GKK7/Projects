import kaggle
import numpy as np
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi()
api.authenticate()
api.competition_download_file("sentiment-analysis-on-movie-reviews","test.tsv.zip",path="./")
api.competition_download_file("sentiment-analysis-on-movie-reviews","train.tsv.zip",path="./")

import zipfile
with zipfile.ZipFile("./test.tsv.zip",'r') as zipref:
    zipref.extractall('./')
with zipfile.ZipFile("./train.tsv.zip",'r') as zipref:
    zipref.extractall('./')

import pandas as pd
df=pd.read_csv("train.tsv",sep="\t")
print(df.head())

df["Sentiment"].value_counts().plot(kind="bar")
plt.show()

seq_len=512
num_samples=len(df)
print(num_samples,seq_len)

from transformers import BertTokenizer
tokenizer=BertTokenizer.from_pretrained("bert-base-cased")

tokens=tokenizer(df["Phrase"].tolist(),max_length=seq_len,truncation=True,padding="max_length",add_special_tokens=True,return_tensors="pt")

print(tokens.keys())

print(tokens["input_ids"])

print(tokens["attention_mask"])

import numpy
with open("movie-xids.npy", "wb") as f:
    np.save(f,tokens["input_ids"].numpy())

with open("movie-xmask.npy", "wb") as f:
    np.save(f,tokens["attention_mask"].numpy())

arr=df["Sentiment"].values
print(arr.shape)

labels=np.zeros((num_samples,arr.max()+1))
labels[np.arange(num_samples),arr]=1
print(labels)

with open('movie-labels.npy', 'wb') as f:
    np.save(f, labels)

with open('movie-xids.npy', 'rb') as f:
    Xids = np.load(f, allow_pickle=True)

with open('movie-xmask.npy', 'rb') as f:
    Xmask = np.load(f, allow_pickle=True)

with open('movie-labels.npy', 'rb') as f:
    labels = np.load(f, allow_pickle=True)

import torch
from torch.utils.data import TensorDataset, DataLoader

input_ids = torch.tensor(Xids, dtype=torch.long)
Xmask = torch.tensor(Xmask, dtype=torch.long)
labels = torch.tensor(labels, dtype=torch.float)

dataset = TensorDataset(input_ids, Xmask, labels)

print(dataset[0])

import torch
from torch.utils.data import Dataset, DataLoader, random_split

class CustomDataset(Dataset):
    def __init__(self, Xids, Xmask, labels):
        self.Xids = Xids
        self.Xmask = Xmask
        self.labels = labels

    def __getitem__(self, index):
        return self.Xids[index], self.Xmask[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

# Create the custom dataset
dataset = CustomDataset(Xids, Xmask, labels)

# Set the batch size and calculate the split index for training and validation datasets
batch_size = 16
split = 0.9
train_size = int(split * len(dataset))
val_size = len(dataset) - train_size

# Split the dataset into training and validation sets
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# Create the DataLoader for training and validation datasets
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# Print the first batch from the training DataLoader
for xid, xmask, label in train_loader:
    print(xid, xmask, label)
    break


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, BertConfig

# Load the pretrained BERT model
bert = BertModel.from_pretrained('bert-base-cased')

# Create a custom PyTorch model with BERT and additional layers
class SentimentClassifier(nn.Module):
    def __init__(self, bert):
        super(SentimentClassifier, self).__init__()
        self.bert = bert
        self.dense = nn.Linear(768, 1024)
        self.relu = nn.ReLU()
        self.output = nn.Linear(1024, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        x = self.dense(cls_token)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x



# Initialize the custom model
model = SentimentClassifier(bert)


# Freeze BERT model parameters
for param in model.bert.parameters():
    param.requires_grad = False

model = SentimentClassifier(bert)

# Define the optimizer, loss function, and evaluation metric
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from tqdm import tqdm  # Add this import at the beginning of your code

# Training loop
# Training loop
epochs = 1

for epoch in range(epochs):
    model.train()
    total_loss = 0

    # Wrap the train_loader with tqdm to show progress
    train_loader_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

    for batch in train_loader_progress:
        optimizer.zero_grad()

        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, torch.argmax(labels, dim=1))

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        # Update the progress bar description with the current batch loss
        train_loader_progress.set_description(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch: {epoch + 1}/{epochs}, Avg Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, torch.argmax(labels, dim=1))

            total_val_loss += loss.item()
            correct += (preds == torch.argmax(labels, dim=1)).sum().item()
            total += labels.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct / total
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Save the model after execution
save_directory = "trained_sentiment_classifier"
torch.save(model.state_dict(), "sentiment_model.pth")

