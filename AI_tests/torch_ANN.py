import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("/home/gkirilov/Documents/pytorch_tutoria/PYTORCH_NOTEBOOKS/Data/iris.csv")

# predict new flower species based on data

# We start off by creating the Model class

class Model(nn.Module):     # Inherit from nn.Module
    def __init__(self, in_features=4,h1=8,h2=9,out_features=3):
        # have at least as many neurons as features
        # how many layers are there
        super().__init__()
        self.fc1=nn.Linear(in_features,h1)  # fully connected layer
        self.fc2=nn.Linear(h1,h2)           # output of 1 leads to 2
        self.out=nn.Linear(h2,out_features) # final output layer

        # Input layer (4 features) -> h1 N-neurons -> h2 N-neurons -> output 3 classes

    def forward(self,x):      # propagation method that propagates forward
        # pass features through the fully connected layer and activation functions
        x=F.relu(self.fc1(x)) # define the activation functions being used - relu and pass first fully connected layer
        x=F.relu(self.fc2(x))
        x=self.out(x)         # final output
        return x

torch.manual_seed(32) # weights and biases in an NN are initialized randomly
model=Model()

# Comment this out to plot only the loss function
#fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
#fig.tight_layout()

plots = [(0,1),(2,3),(0,2),(1,3)]
colors = ['b', 'r', 'g']
labels = ['Iris setosa','Iris virginica','Iris versicolor']

# Comment this out to plot only the loss function
# for i, ax in enumerate(axes.flat):
#     for j in range(3):
#         x = df.columns[plots[i][0]]
#         y = df.columns[plots[i][1]]
#         ax.scatter(df[df['target']==j][x], df[df['target']==j][y], color=colors[j])
#         ax.set(xlabel=x, ylabel=y)
#
# fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0,0.85))
# plt.show()

X = df.drop("target", axis=1)
y = df["target"]

X=X.values
y=y.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=33)

X_train=torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)
y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)
criterion= nn.CrossEntropyLoss()                            # how we measure the error, one hot encoding
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)      # generator object

# Epochs hard to estimate at first
epochs=100
losses=[]

for i in range(epochs):
    # prop Forward and get prediction
    y_pred=model.forward(X_train)

    # calculate loss
    loss=criterion(y_pred, y_train)

    losses.append(loss.detach().numpy())

    if i%10==0:
        print(f"Epoch {i} and loss is {loss}")

    # backprop

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# plot epochs vs loss to see if they are enough

plt.plot(range(epochs),losses)
plt.ylabel("Loss")
plt.xlabel("Epoch")
# plt.show()

# Part 3: Validate the model on the test set

with torch.no_grad():       # deactivate autogradient engine and speeds it up
    y_eval=model.forward(X_test)    # y-eval are the predictions of the test set
    loss = criterion(y_eval,y_test)

print(loss)

correct=0
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_val=model.forward(data)
        # the biggest y_test[i] indicates which class the prediction should belong to
        print(f"{i+1}.) {str(y_val)} {y_test[i]}")  # y-val is a tensor so you need to include the str func

        if y_val.argmax().item()==y_test[i]:
            correct+=1

print(f"We got {correct} correct")

torch.save(model.state_dict(),"My_iris_model_test.pt")
# save the model
new_model=Model()
new_model.load_state_dict(torch.load("My_iris_model_test.pt"))
print(new_model.eval())

mystery_iris=torch.tensor([5.6,3.7,2.2,0.5])

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
fig.tight_layout()

plots = [(0, 1), (2, 3), (0, 2), (1, 3)]
colors = ['b', 'r', 'g']
labels = ['Iris setosa', 'Iris virginica', 'Iris versicolor', 'Mystery iris']

for i, ax in enumerate(axes.flat):
    for j in range(3):
        x = df.columns[plots[i][0]]
        y = df.columns[plots[i][1]]
        ax.scatter(df[df['target'] == j][x], df[df['target'] == j][y], color=colors[j])
        ax.set(xlabel=x, ylabel=y)

    # Add a plot for our mystery iris:
    ax.scatter(mystery_iris[plots[i][0]], mystery_iris[plots[i][1]], color='y')

fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0, 0.85))
# plt.show()
# final prediction on what is the mystery iris
with torch.no_grad():
    print(new_model(mystery_iris))
    print()
    print(labels[new_model(mystery_iris).argmax()])