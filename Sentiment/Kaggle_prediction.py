import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pandas as pd

# Custom SentimentClassifier model
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

# Load the pretrained model
bert = BertModel.from_pretrained('bert-base-cased')
model = SentimentClassifier(bert)
model.load_state_dict(torch.load('sentiment_model.pth'))
model.eval()

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Prepare the data
def prep_data(text):
    tokens = tokenizer.encode_plus(text, max_length=512,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_token_type_ids=False,
                                   return_tensors='pt')
    return tokens

# Perform prediction
with torch.no_grad():
    probs = model(**prep_data("uhhhhh I hate this!!"))[0]

# Get the predicted class
predicted_class = torch.argmax(probs).item()

# Load the data
df = pd.read_csv('test.tsv', sep='\t')
print(df.head())

# Remove duplicates
df = df.drop_duplicates(subset=['SentenceId'], keep='first')
print(df.head())

# Add a new column for predictions
df['PredictedSentiment'] = None

# Prediction function
def predict_sentiment(phrase):
    with torch.no_grad():
        tokens = prep_data(phrase)
        probs = model(**tokens)[0]
        probs = torch.softmax(probs, dim=-1)
        pred = torch.argmax(probs).item()
    return pred

# Add predictions to the DataFrame
df['Sentiment'] = df['Phrase'].apply(predict_sentiment)

# Print the DataFrame
print(df.head())
print(df.tail())