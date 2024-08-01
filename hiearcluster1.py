# Ward's Method: Produces compact, well-separated clusters.
# Complete Linkage: Clusters are compact, but may show irregular shapes if there are long chains.
# Average Linkage: Clusters are balanced and moderately compact.
# Single Linkage: May produce clusters that are linked through chains or long connections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import seaborn as sns


data=pd.read_csv('malldata.csv')
datas=pd.DataFrame(data)
print(datas.head())

x=data[['Annualincome', 'Spendingscore']]
print(x.head())

scaler=StandardScaler()
x=scaler.fit_transform(x)

cluster=AgglomerativeClustering(n_clusters=5,linkage='ward')

predection=cluster.fit_predict(x)

data['cluster']=predection

plt.figure(figsize=(8,7))
# plt.scatter(x=data['Annual Income (k$)'], y=data['Spending Score (1-100)'],hue=data=['cluster'], palette='viridis',s=100)
sns.scatterplot(data=data, x='Annualincome', y='Spendingscore', hue='cluster', palette='viridis', s=100)
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
                                                    
