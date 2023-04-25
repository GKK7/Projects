import flair
import torch
# import model
model_name='ProsusAI/finbert'
from transformers import BertForSequenceClassification
model=BertForSequenceClassification.from_pretrained(model_name)

# convert text into tokens that the model understands
from transformers import BertTokenizer

tokenizer=BertTokenizer.from_pretrained(model_name)

# 1. Tokenize text
# 2. Feed token IDs into model
# 3. Model activations convert into probabilities with Softmax
# 4. Take argmax of those probabilities

# this is our example text
txt = ("Given the recent downturn in stocks especially in tech which is likely to persist as yields keep going up, "
       "I thought it would be prudent to share the risks of investing in ARK ETFs, written up very nicely by "
       "[The Bear Cave](https://thebearcave.substack.com/p/special-edition-will-ark-invest-blow). The risks comes "
       "primarily from ARK's illiquid and very large holdings in small cap companies. ARK is forced to sell its "
       "holdings whenever its liquid ETF gets hit with outflows as is especially the case in market downturns. "
       "This could force very painful liquidations at unfavorable prices and the ensuing crash goes into a "
       "positive feedback loop leading into a death spiral enticing even more outflows and predatory shorts.")

# Bert would be expecting to see 512 tokens in each input
# If the input string contains more than 512 tokens we cut them up to 512 and not include them
tokens=tokenizer.encode_plus(txt, max_length=512,truncation=True,padding="max_length", add_special_tokens=True, return_tensors='pt')

print(tokens)

# Special tokens:
# [CLS]:101 - start, [SEP]:102 end, [MASK]:103, [UNK]:100, [PAD]:0

output=model(**tokens)
print(output[0])

import torch.nn.functional as F
probs=F.softmax(output[0], dim=-1)
print(probs)
pred=torch.argmax(probs)
print(pred.item())