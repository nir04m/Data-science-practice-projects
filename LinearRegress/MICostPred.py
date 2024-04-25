# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:22:24 2024

@author: Oghale Enwa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

raw_data = pd.read_csv("medical_insurance.csv")
medical_data = raw_data.copy()

plt.figure(figsize=(10, 6))
sns.histplot(medical_data['charges'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Insurance Charges')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.show()
X = medical_data.drop('charges', axis=1)
y = medical_data['charges']

dummies = pd.get_dummies(medical_data)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
cat_feat = ['sex','smoker','region']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([('one_hot', one_hot, cat_feat)], remainder='passthrough')

transformered_x = transformer.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(transformered_x, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

# Fit the classifier to the training data
model.fit(X_train, y_train)

# making predictions on the testing set
#y_pred = model.predict(X_test)

rfc = RandomForestRegressor()

# Fit the classifier to the training data
rfc.fit(X_train, y_train)

# making predictions on the testing set
y_pred = rfc.predict(X_test)

model.score(X_test,y_test)
rfc.score(X_test,y_pred)






