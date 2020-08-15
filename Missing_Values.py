#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Pre-processing
#1. Dealing with missing values


# In[ ]:


missing_values_count = df.isnull().sum()


# In[ ]:


missing_values_count[0:100]


# In[ ]:


# how many total missing values do we have?
total_cells = np.product(df.shape)
total_missing = missing_values_count.sum()


# In[ ]:


# percent of data that is missing
(total_missing/total_cells) * 100


# In[ ]:


#Only 8% of data if missing from the dataset
#Now looking at different columns with missing values


# In[ ]:


df.info()


# In[ ]:


#There are total 891 rows of which Age shows 714 values , Embarked show 889 values and Cabin has lot of missing values.
#Object data type is non-numeric , hence one-hot encoding


# In[ ]:


#To check the type of columns
df.dtypes.value_counts()


# In[ ]:


#To represent the number of rows by the number of columns
df.shape


# In[ ]:


# returns the unique value for each variable
df.nunique(axis=0)


# In[ ]:


#To get the scatter plot pairwise
import seaborn as sns
sns.pairplot(df)


# In[ ]:


#Dropping colums which are not much relevant like Name, Ticked and Cabin
cols = ['Name', 'Ticket', 'Cabin']
df1 = df.drop(cols, axis=1)


# In[ ]:


# just how much data did we lose?
print("Columns in original dataset: %d \n" % df.shape[1])
print("Columns with na's dropped: %d" % df1.shape[1])


# In[ ]:


df1['Age'].fillna(df1['Age'].mean(), inplace=True)


# In[ ]:


#df1['Age'].fillna(df1['Age'].median(), inplace=True)


# In[ ]:


for column in df1[['Embarked']]:
    mode = df1[column].mode()
    df1[column] = df1[column].fillna(mode)


# In[ ]:


df1.dropna()


# In[ ]:




