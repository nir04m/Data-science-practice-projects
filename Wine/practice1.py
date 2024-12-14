# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 08:38:40 2024

@author: Oghale Enwa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv('winequality-red.csv')
wine_df = raw_data.copy()
#print(wine_df.isnull().values.any())
#print(wine_df.isnull().sum())

# Select one physicochemical property and quality
property_of_interest = 'volatile acidity'  # Change this to the property you're interested in
quality_column = 'quality'

# Plot the relationship between the selected property and quality
#plt.figure(figsize=(10, 6))
#plt.scatter(wine_df[property_of_interest], wine_df[quality_column], alpha=0.5)
#plt.title(f'{property_of_interest.capitalize()} vs. Wine Quality')
#plt.xlabel(property_of_interest.capitalize())
#plt.ylabel('Quality')
#plt.grid(True)

#plt.figure(figsize=(10, 6))
#sns.boxplot(x=quality_column, y=property_of_interest, data=wine_df)
#plt.title(f'Box Plot of {property_of_interest.capitalize()} by Wine Quality')
#plt.xlabel('Quality')
#plt.ylabel(property_of_interest.capitalize())


#SVM
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Assuming your dataframe is named df and it contains features and the target variable

# Split data into features (X) and target variable (y)
X = wine_df.drop('fixed acidity', axis=1)  # Features
y = wine_df['quality']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Support Vector Machine model
svm_model = SVC()  # Linear kernel, you can also use other kernels like 'rbf' or 'poly'
svm_model.fit(X_train_scaled, y_train)

# Predict on testing data
y_pred = svm_model.predict(X_test_scaled)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report for more detailed evaluation
#print(classification_report(y_test, y_pred))


#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import accuracy_score, classification_report

# Assuming your dataframe is named df and it contains features, including the physicochemical property, and the target variable 'quality'

# Select one physicochemical property as a feature and quality as the target variable
#feature_column = 'density'  # Change this to the physicochemical property you want to use as a feature
#target_variable = 'quality'

# Split data into features (X) and target variable (y)
#X = wine_df[[feature_column]]  # Feature
#y = wine_df[target_variable]  # Target variable

# Split data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# Initialize and train logistic regression model
#logreg_model = LogisticRegression()
#logreg_model.fit(X_train_scaled, y_train)

# Predict on testing data
#y_pred = logreg_model.predict(X_test_scaled)

# Evaluate model
#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy:", accuracy)

# Print classification report for more detailed evaluation
#print(classification_report(y_test, y_pred))

