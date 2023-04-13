import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/gkirilov/Documents/pytorch_tutoria/PYTORCH_NOTEBOOKS/Data/income.csv")

# Separate continuous, categorical and label column names
cat_cols = ['sex', 'education', 'marital-status', 'workclass', 'occupation']
cont_cols = ['age', 'hours-per-week']
y_col = ['label']

print(f'cat_cols  has {len(cat_cols)} columns')  # 5
print(f'cont_cols has {len(cont_cols)} columns') # 2
print(f'y_col     has {len(y_col)} column')      # 1

for cat in cat_cols:
    df[cat] = df[cat].astype('category')

# Set the embedding sizes
# Create a variable "cat_szs" to hold the number of categories in each variable.
# Then create a variable "emb_szs" to hold the list of (category size, embedding size) tuples.
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
print(emb_szs)

# Create an array of categorical values
sx = df['sex'].cat.codes.values
ed = df['education'].cat.codes.values
ms = df['marital-status'].cat.codes.values
wc = df['workclass'].cat.codes.values
oc = df['occupation'].cat.codes.values

cats = np.stack([sx,ed,ms,wc,oc], 1)

print(cats[:5])

# Convert cats to a tensor
cats = torch.tensor(cats, dtype=torch.int64)

# Create an array of continuous values
# Create a NumPy array called "conts" that contains a stack of each continuous column.
conts = np.stack([df[col].values for col in cont_cols], 1)
print(conts[:5])

# Convert "conts" to a tensor

conts = torch.tensor(conts, dtype=torch.float)
print(conts.dtype)

# Create a label tensor. Flatten the tensor so it can be passed into the Loss function

y = torch.tensor(df[y_col].values).flatten()

# Create a train and test sets from cats, conts and y

b= 30000
t= 5000

cat_train = cats[:b-t]
cat_test  = cats[b-t:b]
con_train = conts[:b-t]
con_test  = conts[b-t:b]
y_train   = y[:b-t]
y_test    = y[b-t:b]

# Define the TabularModel class

class TabularModel(nn.Module):

    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        # Call the parent __init__
        super().__init__()

        # Set up the embedding, dropout, and batch normalization layer attributes
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)

        # Assign a variable to hold a list of layers
        layerlist = []

        # Assign a variable to store the number of embedding and continuous layers
        n_emb = sum((nf for ni, nf in emb_szs))
        n_in = n_emb + n_cont

        # Iterate through the passed-in "layers" parameter (ie, [200,100]) to build a list of layers
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))

        # Convert the list of layers into an attribute
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):
        # Extract embedding values from the incoming categorical data
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))
        x = torch.cat(embeddings, 1)
        # Perform an initial dropout on the embeddings
        x = self.emb_drop(x)

        # Normalize the incoming continuous data
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)

        # Set up model layers
        x = self.layers(x)
        return x

# Set the random seed

torch.manual_seed(33)

# Create a TabularModel instance
# Create an instance called "model" with one hidden layer containing 50 neurons and a dropout layer p-value of 0.4

model = TabularModel(emb_szs, conts.shape[1], 2, [50], p=0.4)
print(model)

# Define the loss and optimization functions
# Create a loss function called "criterion" using CrossEntropyLoss
# Create an optimization function called "optimizer" using Adam, with a learning rate of 0.001

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model through 300 epochs

import time

start_time = time.time()

epochs = 300
losses = []

# losses.detach().numpy()
for i in range(epochs):
    i += 1
    y_pred = model(cat_train, con_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.detach().numpy())

    # a neat trick to save screen space:
    if i % 25 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'epoch: {i:3}  loss: {loss.item():10.8f}')  # print the last line
print(f'\nDuration: {time.time() - start_time:.0f} seconds')  # print the time elapsed

plt.plot(range(epochs), losses)
plt.ylabel('Cross Entropy Loss')
plt.xlabel('epoch')
#plt.show()

# Evaluate the test set - With torch set to no_grad, pass cat_test and con_test through the trained model.
# Create a validation set called "y_val". Compare the output to y_test using the loss function defined above.

with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = criterion(y_val, y_test)
print(f'CE Loss: {loss:.8f}')

# Calculate the overall percent accuracy. Compare the argmax values of the y_val validation set to the y_test set

rows = len(y_test)
correct = 0

print(f'{"MODEL OUTPUT":26} ARGMAX  Y_TEST')

for i in range(rows):
    print(f'{str(y_val[i]):26} {y_val[i].argmax().item():^7}{y_test[i]:^7}')

    if y_val[i].argmax().item() == y_test[i]:
        correct += 1

print(f'\n{correct} out of {rows} = {100*correct/rows:.2f}% correct')


def test_data(mdl):  # pass in the name of the model
    # INPUT NEW DATA
    age = float(input("What is the person's age? (18-90)  "))
    sex = input("What is the person's sex? (Male/Female) ").capitalize()
    edn = int(input("What is the person's education level? (3-16) "))
    mar = input("What is the person's marital status? ").capitalize()
    wrk = input("What is the person's workclass? ").capitalize()
    occ = input("What is the person's occupation? ").capitalize()
    hrs = float(input("How many hours/week are worked? (20-90)  "))

    # PREPROCESS THE DATA
    sex_d = {'Female': 0, 'Male': 1}
    mar_d = {'Divorced': 0, 'Married': 1, 'Married-spouse-absent': 2, 'Never-married': 3, 'Separated': 4, 'Widowed': 5}
    wrk_d = {'Federal-gov': 0, 'Local-gov': 1, 'Private': 2, 'Self-emp': 3, 'State-gov': 4}
    occ_d = {'Adm-clerical': 0, 'Craft-repair': 1, 'Exec-managerial': 2, 'Farming-fishing': 3, 'Handlers-cleaners': 4,
             'Machine-op-inspct': 5, 'Other-service': 6, 'Prof-specialty': 7, 'Protective-serv': 8, 'Sales': 9,
             'Tech-support': 10, 'Transport-moving': 11}

    sex = sex_d[sex]
    mar = mar_d[mar]
    wrk = wrk_d[wrk]
    occ = occ_d[occ]

    # CREATE CAT AND CONT TENSORS
    cats = torch.tensor([sex, edn, mar, wrk, occ], dtype=torch.int64).reshape(1, -1)
    conts = torch.tensor([age, hrs], dtype=torch.float).reshape(1, -1)

    # SET MODEL TO EVAL (in case this hasn't been done)
    mdl.eval()

    # PASS NEW DATA THROUGH THE MODEL WITHOUT PERFORMING A BACKPROP
    with torch.no_grad():
        z = mdl(cats, conts).argmax().item()

    print(f'\nThe predicted label is {z}')


test_data(model)