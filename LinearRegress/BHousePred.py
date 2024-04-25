# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:34:13 2024

@author: Oghale Enwa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv("Boston_House_price.csv")
df_house = raw_data.copy()


# Histogram of Housing Prices (MEDV)

#sns.histplot(df_house['MEDV'], bins=20, kde=True, color='green')
#plt.title('Distribution of Housing Prices (MEDV)')
#plt.xlabel('Median Housing Price ($1000s)')
#plt.ylabel('Frequency')


# Boxplot of Housing Age (AGE)

#sns.boxplot(df_house['AGE'], color='yellow')
#plt.title('Distribution of  Housing Age (AGE)')
#plt.xlabel('Housing Age in Years')
#plt.ylabel('Age')


# Scatter Plot of Rooms per Dwelling (RM) vs. Housing Prices (MEDV)

#sns.scatterplot(x='RM', y='MEDV', data=df_house, color='lightblue',edgecolor='black', alpha=0.7)
#plt.title('Rooms per Dwelling (RM) vs. Housing Prices (MEDV)')
#plt.xlabel('Average Number of Rooms per Dwelling')
#plt.ylabel('Median Housing Price ($1000s)')

# Scatter Plot of Crime Rate of Housing Area v/s Housing Prices (MEDV)

#sns.scatterplot(x='CRIM', y='MEDV', data=df_house, color='blue', alpha=0.7)
#plt.title('Crime Rate vs. Housing Prices (MEDV)')
#plt.xlabel('Crime Rate')
#plt.ylabel('Median Housing Price ($1000s)')



rad_medv_mean = df_house.groupby('RAD')['MEDV'].mean().reset_index()

# Bar Plot Average House Price By Accessibility of Road Highways

#sns.barplot(x='RAD', y='MEDV', data=rad_medv_mean, color='orange',edgecolor='black')
#plt.title('Average House Price By Accessibility of Road Highways')
#plt.xlabel('Accessibility of Road Highways')
#plt.ylabel('Mean Housing Price ($1000s)')

# Correlation Heatmap

#plt.figure(figsize=(10, 8))
#sns.heatmap(df_house.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
#plt.title('Correlation Heatmap of Boston Housing Features')

# Split the Data 
X = df_house.drop(columns=['MEDV']) #features
y = df_house['MEDV'] #target variable

# Splitting Data for Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2 ,random_state=2529)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# Fit the model on Training dataset
lr.fit(X_train,y_train)

# # Predictions of Linear Regressoion on Testing Data
y_pred_lr = lr.predict(X_test)



# Accuracy Score of Model
from sklearn.metrics import mean_absolute_percentage_error
error = mean_absolute_percentage_error(y_pred_lr,y_test)
print("Accuracy of Linear Regression is :%.2f "%((1 - error)*100),'%')

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor 

# Fit the model on Training dataset
dtr = DecisionTreeRegressor(max_depth=5)
dtr.fit(X_train,y_train)
y_pred_dtr=dtr.predict(X_test)

# Accuracy Score of Model

error = mean_absolute_percentage_error(y_pred_dtr,y_test)
print("Accuracy of Decision Tree Regressor is :%.2f "%((1 - error)*100),'%')

# Random Forest Regressor 
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()

# Fit the model on Training datset
rfr.fit(X_train,y_train)

# Predictions of  Ranforest Forest Regressor on Testing Data
y_pred_rfr = rfr.predict(X_test)

# Accuracy Score of Model

error = mean_absolute_percentage_error(y_pred_rfr,y_test)
print("Accuracy of Random Forest Regressor is :%.2f "%((1 - error)*100),'%')


