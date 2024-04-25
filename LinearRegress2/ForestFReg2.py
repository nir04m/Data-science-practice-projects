# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:53:20 2024

@author: Oghale Enwa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv("covtype.csv")
data = raw_data.copy()

#check mising values
#print(list(data.isnull().any()))

#About Target/Cover_Type variable 
data.Cover_Type.value_counts()

#count plot of target
#sns.countplot(x='Cover_Type', data=data)

#Take some column
col = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points']

train = data[col]

#histogram
#train.hist(figsize=(13, 11))

#Boxplot
#plt.style.use('ggplot')
#for i in col:
#    plt.figure(figsize=(13, 7))
#    plt.title(str(i) + " with " + str('Cover_Type'))
#    sns.boxplot(x=data.Cover_Type, y=train[i])

#Corralation
#plt.figure(figsize=(12, 8))
#corr = train.corr()
#sns.heatmap(corr, annot=True)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

#separate features and target
feature = data.iloc[:, :54] #Features of data
y = data.iloc[:, 54]  #Target of data

# Features Reduction
ETC = ExtraTreesClassifier()
ETC = ETC.fit(feature, y)

model = SelectFromModel(ETC, prefit=True)
X = model.transform(feature) #new features

#Split the data into test and train formate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)



from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100)

#fit
RFC.fit(X_train, y_train)

#prediction
y_pred = RFC.predict(X_test)

#score
print("Accuracy -- ", RFC.score(X_test, y_test)*100)

#confusion
cm = confusion_matrix(y_pred, y_test)
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt='g')

