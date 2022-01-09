#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier


# In[22]:


# Data load/read
df = pd.read_csv("loan_borowwer_data.csv")
print(df.head())


# In[23]:


# Get the independent and dependent datas
X = df.iloc[:, 2:13]
Y = df["not.fully.paid"]

# Split train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 10)


# In[24]:


" As output 'not fully paid' columns has only binary values as a practice we can use "
" ************* Logistic Regression ************* "

LogReg = LogisticRegression()
LogReg.fit(X_train, Y_train)
y_pred = LogReg.predict(X_test)
accuracy = metrics.accuracy_score(y_pred, Y_test)
accuracy


# In[25]:


" ************* Decision Tree Classifier ************* "
DTClass = DecisionTreeClassifier()
DTClass.fit(X_train, Y_train)
y_pred = LogReg.predict(X_test)
accuracy2 = metrics.accuracy_score(y_pred, Y_test)
accuracy2


# In[26]:


" ****** Random Forest Classifier *************"
random_forest = RandomForestClassifier()
random_forest.fit(X_train, Y_train)

# Predict the values based on independent test data
y_pred = random_forest.predict(X_test)

# Measure the accuracy Y vs y_predicted
print("Accuracy = ", metrics.accuracy_score(Y_test, y_pred))

df = pd.DataFrame({'Actual': Y_test, 'Predict': y_pred})


# In[ ]:




