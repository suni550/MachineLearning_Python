#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns


# In[2]:


df = pd.read_csv("FyntraCustomerData.csv")
df.head()


# In[3]:


sns.set_palette("GnBu_d")
sns.set_style('whitegrid')


# In[4]:


sns.jointplot(x = 'Time_on_Website', y = 'Yearly_Amount_Spent', data = df)


# In[5]:


sns.jointplot(x = 'Time_on_App', y = 'Yearly_Amount_Spent', data = df)


# In[6]:



sns.pairplot(df)


# In[7]:


sns.lmplot(x='Length_of_Membership',y='Yearly_Amount_Spent',data=df)


# In[8]:


X = df.iloc[:, 3:7]
#x
y = df['Yearly_Amount_Spent']
y


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)
print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)


# In[10]:


reg = LinearRegression()


# In[11]:


reg.fit(X_train, y_train)


# In[12]:


plt.scatter(y_test, reg.predict(X_test), label = 'TestData')
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')


# Compute Root Mean Sqaure Error
# 

# In[13]:


y_pred =reg.predict(X_test)

#print('RMSE:', round( np.sqrt(metrics.mean_squared_error(y_test, predictions)),2) )
print('RMSE:',  round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),3))


# In[14]:


#reg.coef_
#X.columns

coeffecients = pd.DataFrame(reg.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients

Interpreting the coefficients:

Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 26.08 total dollars spent.
Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 39.18 total dollars spent.
Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.40 total dollars spent.
Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.41 total dollars spent.