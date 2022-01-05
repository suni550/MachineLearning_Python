#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("prisoners.csv")


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.describe()


# In[ ]:


columns_to_sum = list(df)
print(columns_to_sum)
#columns_to_sum.remove('STATE/UT')
#columns_to_sum.remove('YEAR')
list_to_remove = ['STATE/UT', 'YEAR']
columns_to_sum = [elem for elem in columns_to_sum if elem not in list_to_remove]
columns_to_sum


# In[ ]:


# 2. Data Manipulation
" Sum of the elements along the columns axis"
#df['total_beneitted'] = df.iloc[:,2:6].sum(axis = 1)
df['total_beneitted'] = df[columns_to_sum].sum(axis = 1)
#dir(df)
df


# In[ ]:




