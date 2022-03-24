#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np


# In[27]:


data=pd.read_csv("CC GENERAL.CSV")
data=data.drop("CUST_ID", axis=1)
data.head()


# In[28]:


data.info()


# In[29]:


data['MINIMUM_PAYMENTS'].fillna(data['MINIMUM_PAYMENTS'].mean(),inplace=True)
data['CREDIT_LIMIT'].fillna(data['CREDIT_LIMIT'].mean(),inplace=True)


# In[30]:


data.describe()


# In[ ]:





# In[31]:


from sklearn.cluster import AgglomerativeClustering
model=AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
clust_labels=model.fit_predict(data)


# In[32]:


agglomerative=pd.DataFrame(clust_labels)
agglomerative.head()


# In[33]:


import matplotlib.pyplot as plt
fig =plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter (data ['BALANCE'] , data ['PURCHASES'] , c= agglomerative[0], s=50)
ax.set_title("Agglomerative Clutering")
ax.set_xlabel("BALANCE")
ax.set_ylabel("PURCHASES")
plt.colorbar(scatter)


# In[34]:


from sklearn.cluster import KMeans  
kmeans=KMeans(n_clusters=5, random_state=0)  
kmeans.fit(data)


# In[35]:


labels=pd.DataFrame(kmeans.labels_)
labels


# In[36]:


kmeans.predict(data)
print(kmeans.cluster_centers_)


# In[38]:


sum_of_squard_distances = []
k = range(1,15)
for i in k :
    km = kmeans(n_clusters=k)
    km = km.fit(data)
    sum_of_squard_distances.append(km.inertia_)


# In[ ]:




