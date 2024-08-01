# clustering is the process of grouping the similar datas using unlabeld data 
# here the cluster is the collection of similar datas
# Labeled data the data with column_name or attributes
# unlabeld data means the collection of feature or column datas


# in k means clustering we defined the k values based 
# we will receive the inital cenroids randomly
# assige each dat to the nearest centroids
# update the centroids based on the mean of the assigned data points
# repeat the process until the centroids are not changed or minimally changed
# on that it will cluster the data into each groups 

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    'CustomerID': [1, 2, 3, 4, 5],
    'Annual Income': [15000, 16000, 17000, 19000, 21000],
    'Spending Score': [39, 81, 6, 77, 40]
}

df=pd.DataFrame(data)
# print(df.head())

x=df[['Annual Income','Spending Score']]

scaler=StandardScaler()
scaled_x=scaler.fit_transform(x)
# print(scaled_x)

kmean=KMeans(n_clusters=3,random_state=42)
df['Cluster']=kmean.fit_predict(scaled_x)

# print(df['Cluster'])

centers=kmean.cluster_centers_
# print(centers)

# arr=[1,2,3,4]
# print(arr[2,:])

inversed_centers=scaler.inverse_transform(centers)
print(inversed_centers)
# print(inversed_centers[0,:])
#  the : operator is used for mostly in the matrix 
# if u define arr[0,:]  here 0 is the row and it will get all the 
# elementsthe 0th row
#  it will get all the column in the 0 th row

plt.figure(figsize=(10,6))
# this figure is used to chagne the screen of plt graphs nd bars 
plt.scatter(df['Annual Income'],df['Spending Score'],c=df['Cluster'],cmap='viridis',marker='o')
plt.scatter(inversed_centers[:,0],inversed_centers[:,1],c='red',s=300,alpha=0.75,marker='x')
# here marker is the symbol that will be displayed s is the size of the marker
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.show()