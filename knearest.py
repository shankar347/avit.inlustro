import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

n_samples=300,
n_features=2,
n_clusters=3,
random_state=42

x,_=make_blobs(n_samples=n_samples,n_features=n_features,centers=n_clusters,random_state=random_state)

kmeans=KMeans(n_clusters=n_clusters,random_state=random_state)
kmeans.fit(x)

labels=kmeans.predict(x)

centers=kmeans.cluster_centers_

plt.figure(figsize=(6,8))
plt.scatter(x[:,0],x[:,1],c=labels,s=50,marker='o')
plt.scatter(centers[:,0],centers[:,1],c='red',s=200,marker='X',label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
