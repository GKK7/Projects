# Import the required modules. Unsupervised learning with KMeans
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import warnings
import matplotlib.pyplot as plt

# FutureWarning error from sklearn - ignore
warnings.simplefilter(action='ignore', category=FutureWarning)

# Get the data
data = pd.read_excel('Real Estate.xlsx')

# Remove the last row, which is totals, to not skew results
data = data.drop([113])

# Provide two dimensional dataset
x = data[["Общо вписвания", "Възбрани"]]

# 4 clusters selected after some tests
kmeans = KMeans(n_clusters=4)
kmeans.fit(x)
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Plot and visualize the data with matplotlib
plt.scatter(x['Общо вписвания'], x['Възбрани'], c=labels.astype(float), s=20, alpha=0.5)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=20)
plt.title("Real Estate data")
plt.xlabel("Общо вписвания")
plt.ylabel("Възбрани")

plt.show()

# Find a way to display annotations on the scatter plot
