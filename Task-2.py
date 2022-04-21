#!/usr/bin/env python
# coding: utf-8

# # GRIP:- The Spark Foundations
# ## Data Science and Business Analytics - 
# ## Internship Creator:- A.DEEPTHI PRIYANKA 
# ## Task-2: Prediction using unsupervised ML 
# ## 2.From the given dataset  "Iris" we have to predict the optimum number of clusters and represent the data visually.

# In[7]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df.head() # See the first 5 rows


# In[8]:


data.drop(['Species','Id'],axis=1)


# In[9]:


x = data.iloc[:, [0, 1, 2, 3]].values


# In[10]:


from sklearn.cluster import KMeans


# In[14]:


# Finding the optimum number of clusters for k-means classification

x = iris_df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# Plotting the results onto a line graph, 
# `allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# In[12]:


km = KMeans(n_clusters=3,max_iter=300,n_init=10)
y_kmeans = km.fit_predict(x)


# In[13]:


y_kmeans


# In[10]:


plt.scatter(x[y_kmeans == 0,0], x[y_kmeans == 0,1],s=100,c = 'red',label = 'Iris-setosa')

plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1,1],s=100,c = 'blue',label = 'Iris-versicolour')

plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1],s=100,c = 'green',label = 'Iris-virginica')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1], s=100,c='yellow',label='Centroids')

plt.rcParams["figure.figsize"]=10,8


# In[ ]:




