#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# In[ ]:


df = pd.read_csv("voice.csv")
df
df.label


# In[ ]:


df["label"] = df["label"].map({"male": 0, "female":1})
#df["label"] = df["label"].map({"male": 0, "female":1})
#data['dis']=data['diagnosis'].map({'M':1,'B':0})
df


# In[ ]:


Train, Test = train_test_split(df, test_size = 0.20, random_state = 1)

X_train = Train.iloc[:, :-1]
X_train
Y_train = Train['label']
Y_train


# In[ ]:


X_test = Train.iloc[:, :-1]
X_test
Y_test = Train['label']
Y_test


# In[ ]:


logistic = LogisticRegression(solver='lbfgs', max_iter=1000)
logistic.fit(X_train,Y_train)


# In[ ]:


# Predict based on the model
y_predict = logistic.predict(X_test)

#Find accuracy
df_predict = pd.DataFrame({'Actual': Y_test, 'Predict': y_predict})
df_predict


# In[ ]:


accuracy = metrics.accuracy_score(y_predict, Y_test)
accuracy


# In[106]:


corr = df.corr()
plt.figure(figsize = (12,12))
sns.heatmap(corr, cbar= True, square = True, cmap = 'coolwarm', center = 0)
plt.show()
corr


# In[114]:


"After careful observation, the following variables having a high correlation with 'label'"
"Between the independent variables centroid and meanfreq there is a strong correlation: there is multi colinearility"
"removing one of them does not have impact on the output"
#prediction_var = ['meanfreq', 'sd', 'sp.ent', 'meanfun', 'IQR'] # with these var getting accuracy of 90% 
#prediction_var = ['meanfreq', 'sd', 'sp.ent','meanfun' ] # with these getting accuracy of 89.9% 
prediction_var = ['meanfreq', 'sd', 'sp.ent'] # with these var, getting accuracy of 73.1%


# In[115]:


Train_2, Test_2 = train_test_split(df, test_size = 0.20, random_state = 1)

X_train_2 = Train_2[prediction_var]
X_train_2
Y_train_2 = Train_2['label']
Y_train


# In[116]:


X_test_2 = Test_2[prediction_var]
X_test_2
Y_test_2 = Test_2['label']
Y_test_2


# In[117]:


logisticReg = LogisticRegression(solver='lbfgs', max_iter=1000)
logisticReg.fit(X_train_2, Y_train_2)


# In[118]:


y_predict_2 = logisticReg.predict(X_test_2)


# In[112]:


df_predict_2 = pd.DataFrame({'Actual': Y_test_2, 'Predict': y_predict_2})
df_predict_2


# In[119]:


# Accuracy Score 
acc_score_2 = metrics.accuracy_score(y_predict_2, Y_test_2)
acc_score_2


# In[ ]:




