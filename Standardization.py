#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#One hot encoding
dummies = []
cols = ['Pclass', 'Sex', 'Embarked']
for col in cols:
    dummies.append(pd.get_dummies(df[col]))
titanic_dummies = pd.concat(dummies, axis=1)
df1 = pd.concat((df1,titanic_dummies), axis=1)


# In[ ]:


df1.head()


# In[ ]:


df1.info()


# In[ ]:


pip install mlxtend


# In[ ]:


# 3.Normalisation
# The fare and age may the impact differently. To have their impact on dependent variable of these two attributes equally, normalisation is done so that they fall in the range of 0-1 having equal impact on our target variable.


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (-3,3))
scaler.fit_transform(df.Fare.values.reshape(-1,1))


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit_transform(df.Age.values.reshape(-1,1))


# In[ ]:


# for Box-Cox Transformation
from scipy import stats
# for min_max scaling
import mlxtend
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


positive_original_data = df.Fare>0
positive_fare = df.Fare.loc[positive_original_data]
normalized_data = stats.boxcox(positive_fare)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_fare, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")


# In[ ]:


positive_original_data = df.Age>0
positive_fare = df.Age.loc[positive_original_data]
normalized_data = stats.boxcox(positive_fare)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_fare, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")


# In[ ]:


col_names = ['Age','Fare']
features = df[col_names]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)


# In[ ]:


scaled_features = pd.DataFrame(features, columns = col_names)


# In[ ]:


scaled_features.head()


# In[ ]:




