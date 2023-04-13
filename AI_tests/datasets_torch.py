import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Loader to feed in batches into it

# Method 1

df=pd.read_csv("/home/gkirilov/Documents/pytorch_tutoria/PYTORCH_NOTEBOOKS/Data/iris.csv")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
fig.tight_layout()

plots = [(0,1),(2,3),(0,2),(1,3)]
colors = ['b', 'r', 'g']
labels = ['Iris setosa','Iris virginica','Iris versicolor']

for i, ax in enumerate(axes.flat):
    for j in range(3):
        x = df.columns[plots[i][0]]
        y = df.columns[plots[i][1]]
        ax.scatter(df[df['target']==j][x], df[df['target']==j][y], color=colors[j])
        ax.set(xlabel=x, ylabel=y)

fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0,0.85))
#plt.show()

from sklearn.model_selection import train_test_split

features=df.drop("target",axis=1).values        #remove the values from the target column
label=df["target"].values
X_train, X_test, y_train, y_test = train_test_split(features,label,test_size=0.2,random_state=33)

print(X_train)      # the result is a numpy array that we want to convert to a tensor

X_train=torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)
y_train=torch.LongTensor(y_train).reshape(-1,1)
y_test=torch.LongTensor(y_test).reshape(-1,1)
print(y_test)

# Method 2
from torch.utils.data import TensorDataset, DataLoader
data=df.drop("target",axis=1).values
labels=df["target"].values
iris=TensorDataset(torch.FloatTensor(data),torch.LongTensor(labels))

#for i in iris:
    #print(i)

# Loader object that can shuffle the data and produce batches of the data, as we don't want
# to use the full dataset at once, but instead use a batch of the data. Data shuffle and loader

iris_loader=DataLoader(iris,batch_size=50, shuffle=True)

# print 3x50 batches from the 150 total examples.
for i_batch, sample_batch in enumerate(iris_loader):
    print(i_batch,sample_batch)