#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


#pip install pandas_profiling


# # New Section

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


df = pd.read_csv("heart-disease.csv")


# In[12]:


df.head()


# In[13]:


fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(df.corr(), square=True, annot=True, cmap=plt.cm.RdYlGn, vmin=-1, vmax=1, ax=ax)
ax.set_title('Correlation between numerical variables', fontsize=18)


# In[14]:


df.describe(include="all")


# In[15]:


#Using Profile Report for EDA
from pandas_profiling import ProfileReport
ProfileReport(df)


# In[8]:


get_ipython().system('pip install sweetviz')


# In[16]:


#Using sweetviz for eda
import sweetviz as sv
my_report = sv.analyze(df)
my_report.show_html()

