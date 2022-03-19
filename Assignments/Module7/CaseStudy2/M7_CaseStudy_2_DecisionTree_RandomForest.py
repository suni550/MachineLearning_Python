#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np 

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


horses = pd.read_csv("horse.csv")
horses.info()

horses.head()

target = horses.outcome
horses.drop('outcome', axis =1, inplace = True )

horses.isnull().sum()

df_cat = horses[['surgery', 'age', 'temp_of_extremities','peripheral_pulse',
       'mucous_membrane', 'capillary_refill_time', 'pain', 'peristalsis',
       'abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux', 'rectal_exam_feces', 'abdomen',
      'abdomo_appearance', 'surgical_lesion','cp_data']]

df_encoded = pd.get_dummies(data = df_cat)

df_encoded.head()

df_numerical = horses[['rectal_temp', 'pulse', 'respiratory_rate', 'nasogastric_reflux_ph', 'packed_cell_volume', 
                       'total_protein', 'abdomo_protein', 'lesion_1', 'lesion_2', 'lesion_3']]

# Concatinate the data frames
horses_new = pd.concat([df_numerical, df_encoded], axis = 1, join = 'outer')
horses_new


impute = SimpleImputer(missing_values= np.nan, strategy="most_frequent")
X = impute.fit_transform(horses_new)
X

y= target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)


# build DecisionTree Classifier
DTreeClass = DecisionTreeClassifier()

DTreeClass.fit(X_train, y_train)
y_predict = DTreeClass.predict(X_test)


DTaccuracy = accuracy_score(y_predict,y_test)
print("DTaccuracy = ", DTaccuracy)


# build RandomForest
RFClassifier = RandomForestClassifier()
RFClassifier.fit(X_train, y_train)

y_pred = RFClassifier.predict(X_test)

RFaccuracy = metrics.accuracy_score(y_pred, y_test)
print("RFaccuracy = ", RFaccuracy)