#!/usr/bin/env python
# coding: utf-8

# In[134]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn import tree
#from sklearn import preprocessing


# In[135]:


horses_df = pd.read_csv("horse.csv")
horses_df.info()

'based on the outcome of this we can say there are some missing values'
'e.g., surgery column has total 299 values (not missed any values)'
'e.g., "pain" column has only 244 values means reaming 55 values are missing'


# In[136]:


horses_df.isnull()


# In[137]:


target = horses_df.outcome
horses_df.drop('outcome', axis = 1, inplace = True)
horses_df.head()


# In[138]:


category_variables = ['surgery', 'age', 'temp_of_extremities','peripheral_pulse',
       'mucous_membrane', 'capillary_refill_time', 'pain', 'peristalsis',
       'abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux', 'rectal_exam_feces', 'abdomen',
      'abdomo_appearance', 'surgical_lesion','cp_data']

for category in category_variables:
    horses_df[category] = pd.get_dummies(horses_df[category])

horses_df


# In[139]:


X = horses_df
y = target
print(type(X))


# In[140]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =1)
print(type(X_train))


# In[141]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values= np.nan,strategy="most_frequent")
X_train = imp.fit_transform(X_train)
X_test = imp.fit_transform(X_test)
print(type(X_train))


# In[142]:


from sklearn.tree import DecisionTreeClassifier
DTclassifier = DecisionTreeClassifier()
DTclassifier.fit(X_train,y_train)


# In[143]:


y_predict = DTclassifier.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_predict,y_test)
print(accuracy)


# In[152]:



from sklearn.ensemble import RandomForestClassifier
RFClassifier = RandomForestClassifier()
RFClassifier.fit(X_train, y_train)

y_pred = RFClassifier.predict(X_test)

accuracy = metrics.accuracy_score(y_pred, y_test)
accuracy


# In[ ]:




