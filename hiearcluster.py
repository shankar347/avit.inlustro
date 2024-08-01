# herarchial clustering is a type of clustering
# it can be divided into two types
# 1 bottom up (agglomerative)approach
# 2 top down (divisive) approach

#  in buttomup or agglomerative appraoch 

# we starting each datapoint as a seperate cluster and merge them
# untill it form to one cluster or the stopping criterion mets


# in topdonw or divisive approach

# we starting with considering all the datapoints as a single cluster
# and split them until each datapoints have it's owns cluster or 
# stopping criterion mets

# key concepts in agglomerative clustering
#   distance matrix
    #  it is like the general square matrix that contins the distance between
# all the data in the datas .it quantifies how each data is close or away from
# other data
#   linkage matrix
#    that contains the information about the hierarchical clustering
# it records how clusters are merged togther

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

x,y=make_blobs(n_samples=100,centers=3,cluster_std=0.60,random_state=0)

# print(x)
# print(y)

hi_cluster=AgglomerativeClustering(n_clusters=3)
print(hi_cluster.fit_predict(x))

label=hi_cluster.labels_

plt.scatter(x[:,0],x[:,1],c=label,cmap='viridis')
# here the cmap is used to provide the color
# here label is used to address the cluster by providing the colors

plt.title('Agglomerative Hierarchical Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()