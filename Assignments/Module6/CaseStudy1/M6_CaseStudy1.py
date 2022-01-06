#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import matplotlib.pyplot as plot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


df = pd.read_csv("prisoners.csv")


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.describe()


# In[14]:


sum_of_cols = list(df)
del_columns = ['STATE/UT', 'YEAR']

sum_of_cols = [item for item in sum_of_cols if item not in del_columns]
sum_of_cols


# In[15]:


'Create a new column -’total_benefitted’ that is a sum of inmates benefitted through all modes.'

df['total_benefitted'] = df[sum_of_cols].sum(axis = 1)
#df


# In[16]:


sum_of_cols.append('total_benefitted')
sum_of_cols


# In[17]:


'Create a new row - “totals” that is the sum of all inmates benefitted through each mode across all states.'
#df_xtemp = df[sum_of_cols].sum(axis = 0)
#df_totals = df.append(df_xtemp, ignore_index=True)

df_totals = df.append(df[sum_of_cols].sum(axis = 0), ignore_index=True)
df_totals


# In[ ]:




