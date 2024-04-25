# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:07:46 2024

@author: Oghale Enwa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv("FuelConsumption.csv")
fuel = raw_data.copy()

#sns.histplot(fuel['FUEL CONSUMPTION'], bins=20, kde=True, color='green')
#plt.title('Distribution for fuel consumption')
#plt.xlabel('Average Fuel consumption')
#plt.ylabel('Frequency')

#sns.histplot(x='COEMISSIONS ', y='MAKE', data=fuel, color='blue', alpha=0.7)
#plt.title('Car make  vs. emission')
#plt.xlabel('Car make')
#plt.ylabel('Car coemissions')

#sns.scatterplot(x='COEMISSIONS ', y='ENGINE SIZE', data=fuel, color='blue', alpha=0.7)
#plt.title('emission  vs. engine size')
#plt.xlabel('Car emission')
#plt.ylabel('Car engine size')


#plt.scatter(fuel['FUEL CONSUMPTION'],fuel['COEMISSIONS '],color = 'green')
#plt.xlabel('FUEL CONSUMPTION')
#plt.ylabel("EMISSION")

#plt.scatter(fuel['ENGINE SIZE'],fuel['COEMISSIONS '],color = 'green')
#plt.xlabel('ENGINE SIZE')
#plt.ylabel("EMISSION")

categorical_cols = ["MAKE", "MODEL", "VEHICLE CLASS", "TRANSMISSION", "FUEL"]
encoded_data = pd.get_dummies(fuel, columns=categorical_cols)

X = encoded_data.drop(columns=["COEMISSIONS "]) #features
y = encoded_data['COEMISSIONS '] #target variable

# Splitting Data for Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2 ,random_state=42)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# Fit the model on Training dataset
lr.fit(X_train,y_train)

# # Predictions of Linear Regressoion on Testing Data
y_pred_lr = lr.predict(X_test)

from sklearn.metrics import mean_absolute_percentage_error
error = mean_absolute_percentage_error(y_pred_lr,y_test)
print("Accuracy of Linear Regression is :%.2f "%((1 - error)*100),'%')


#from sklearn.ensemble import RandomForestRegressor
#rfr = RandomForestRegressor()

# Fit the model on Training datset
#rfr.fit(X_train,y_train)

# Predictions of  Ranforest Forest Regressor on Testing Data
#y_pred_rfr = rfr.predict(X_test)

# Accuracy Score of Model
#error = mean_absolute_percentage_error(y_pred_rfr,y_test)
#print("Accuracy of Random Forest Regressor is :%.2f "%((1 - error)*100),'%')


#from sklearn.svm import SVR
#svr = SVR()

# Fit the model on Training datset
#svr.fit(X_train,y_train)

# Predictions of  Ranforest Forest Regressor on Testing Data
#y_pred_svr = svr.predict(X_test)

# Accuracy Score of Model
#error = mean_absolute_percentage_error(y_pred_svr,y_test)
#print("Accuracy of Support Vector Regressor is :%.2f "%((1 - error)*100),'%')


#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF
# Define the kernel (RBF kernel)
#kernel = RBF()
#gp_regressor = GaussianProcessRegressor(kernel=kernel)

# Fit the model on Training datset
#gp_regressor.fit(X_train,y_train)

# Predictions of  Ranforest Forest Regressor on Testing Data
#y_pred_gp = gp_regressor.predict(X_test)

# Accuracy Score of Model
#error = mean_absolute_percentage_error(y_pred_gp,y_test)
#print("Accuracy of Gaussian Process Regressor is :%.2f "%((1 - error)*100),'%')


plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_train)), y_train, color='blue', label='y_train')
plt.xlabel('Index')
plt.ylabel('Target Variable')
plt.title('Plot of y_train')
plt.legend()




